from sys import dont_write_bytecode
from numpy import einsum, float32, tile
from regex import B
import torch
import math
import einops
class flash_attention_pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        device = q.device
        B, N_q, d_model = q.shape
        _, N_k, _ = k.shape
        tile_size = 64
        T_q = math.ceil(N_q / tile_size)
        T_k = math.ceil(N_k / tile_size)
        O = torch.empty((B, N_q, d_model), device=device, dtype=torch.float32)
        L = torch.empty((B, N_q), device=device, dtype=torch.float32)
        for batch in range(B):
            Q_b, K_b, V_b = q[batch], k[batch], v[batch]
            for i in range(T_q):
                q_start, q_end = i*tile_size, min((i+1)*tile_size, N_q)
                tiled_q_n = q_end - q_start
                Q_i = Q_b[q_start:q_end, :]
                O_i = torch.zeros((tiled_q_n, d_model), device=device, dtype=torch.float32)
                l_i = torch.zeros((tiled_q_n,), device=device, dtype=torch.float32)
                m_i = torch.full((tiled_q_n,),float('-inf'), device=device, dtype=torch.float32)
                
                for j in range(T_k):
                    k_start, k_end = j*tile_size, min((j+1)*tile_size, N_k)
                    tiled_k_n = k_end - k_start
                    K_j, V_j = K_b[k_start:k_end, :], V_b[k_start:k_end, :]
                    S_i_j = einops.einsum(Q_i, K_j, 'q_tile d, k_tile d -> q_tile k_tile')
                    S_i_j *= d_model**(-0.5)
                    if is_causal:
                        diag =  q_start - k_start#q_start+i >= k_start+j -> j - i <= q_start - k_start (对角线的计算方式：j - i<= diagonal)
                        causal_mask = torch.tril(torch.ones((tiled_q_n, tiled_k_n), dtype=torch.bool, device=device), diagonal=diag,  )
                        S_i_j = torch.masked_fill(S_i_j, ~causal_mask, float('-inf'))
                    #online softmax
                    m_i_new = torch.maximum(m_i, S_i_j.max(1).values)
                    P_i = torch.exp(S_i_j-m_i_new.unsqueeze(-1))
                    l_i = torch.exp(m_i - m_i_new)*l_i + torch.sum(P_i, dim=1)
                    O_i = torch.diag(torch.exp(m_i - m_i_new))@O_i + P_i@V_j
                    m_i = m_i_new

                O_i = torch.diag(1.0 / l_i)@O_i
                l_i = m_i + torch.log(l_i)
                O[batch, q_start:q_end, :] = O_i.to(q.dtype)
                L[batch, q_start:q_end] = l_i
        ctx.save_for_backward(q, k, v, O, L)
        ctx.is_causal = is_causal
        return O


    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # 从ctx获取saved tensors
        q, k, v, O, L = ctx.saved_tensors
        device = grad_out.device

        # 初始化梯度
        dQ = torch.zeros_like(q)
        dK = torch.zeros_like(k)
        dV = torch.zeros_like(v)

        # 获取维度
        batch_size, N_q, d_model = q.shape
        _, N_k, _ = k.shape

        # 设置tile size
        B_q, B_k = 64, 64
        T_q = math.ceil(N_q / B_q)
        T_k = math.ceil(N_k / B_k)
        scale = d_model**(-0.5)

        # 对每个batch进行处理
        for batch in range(batch_size):
            Q_b = q[batch]  # (N_q, d_model)
            K_b = k[batch]  # (N_k, d_model)
            V_b = v[batch]  # (N_k, d_model)
            O_b = O[batch]  # (N_q, d_model)
            L_b = L[batch]  # (N_q,)
            dO_b = grad_out[batch]  # (N_q, d_model)

            # 计算 D = rowsum(O ◦ dO)
            D_b = torch.sum(O_b * dO_b, dim=-1)  # (N_q,)

            # 对每个key tile
            for j in range(T_k):
                K_start = j * B_k
                K_end = min(N_k, (j+1) * B_k)
                tiled_k_n = K_end - K_start

                K_j = K_b[K_start:K_end, :]  # (tiled_k_n, d_model)
                V_j = V_b[K_start:K_end, :]  # (tiled_k_n, d_model)

                dK_j = torch.zeros((tiled_k_n, d_model), dtype=torch.float32, device=device)
                dV_j = torch.zeros((tiled_k_n, d_model), dtype=torch.float32, device=device)

                # 对每个query tile
                for i in range(T_q):
                    Q_start = i * B_q
                    Q_end = min(N_q, (i+1) * B_q)
                    tiled_q_n = Q_end - Q_start

                    Q_i = Q_b[Q_start:Q_end, :]  # (tiled_q_n, d_model)
                    dO_i = dO_b[Q_start:Q_end, :]  # (tiled_q_n, d_model)
                    L_i = L_b[Q_start:Q_end]  # (tiled_q_n,)
                    D_i = D_b[Q_start:Q_end]  # (tiled_q_n,)

                    # 计算 S = QK^T / sqrt(d)
                    S_ij = Q_i @ K_j.T * scale  # (tiled_q_n, tiled_k_n)

                    # Causal masking
                    if ctx.is_causal:
                        q_indices = torch.arange(Q_start, Q_end, device=device)[:, None]
                        k_indices = torch.arange(K_start, K_end, device=device)[None, :]
                        mask = q_indices >= k_indices
                        S_ij = torch.where(mask, S_ij, torch.tensor(float('-inf'), device=device))

                    # 计算 P = exp(S - L)
                    P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))  # (tiled_q_n, tiled_k_n)

                    # dV += P^T @ dO
                    dV_j += P_ij.T @ dO_i

                    # dP = dO @ V^T
                    dP_ij = dO_i @ V_j.T  # (tiled_q_n, tiled_k_n)

                    # dS = P ◦ (dP - D) / sqrt(d)
                    dS_ij = P_ij * (dP_ij - D_i.unsqueeze(-1)) * scale  # (tiled_q_n, tiled_k_n)

                    # dQ += dS @ K
                    dQ[batch, Q_start:Q_end, :] += dS_ij @ K_j

                    # dK += dS^T @ Q
                    dK_j += dS_ij.T @ Q_i

                dK[batch, K_start:K_end, :] = dK_j
                dV[batch, K_start:K_end, :] = dV_j

        return dQ, dK, dV, None  # None for is_causal







        