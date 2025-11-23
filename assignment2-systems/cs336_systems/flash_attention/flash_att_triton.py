import triton
import torch
import math
import triton.language as tl

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    scaler = scale
    o_i = tl.zeros((Q_TILE_SIZE,D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    Q_i = tl.load(Q_block_ptr, boundary_check=(0,1))
    T_k = tl.cdiv(N_KEYS, K_TILE_SIZE)

    for j in range(T_k):
        K_block_ptr = tl.make_block_ptr(
            K_ptr + stride_kb * batch_index,
            shape=(D, N_KEYS),
            strides=(stride_kd, stride_kk),
            offsets=(0, j * K_TILE_SIZE),
            block_shape=(D, K_TILE_SIZE),
            order=(0, 1)
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + stride_vb * batch_index,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(j * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0)
        )
        #有了k、v地址就可以load进来了
        K_j = tl.load(K_block_ptr, boundary_check=(1,0))
        V_j = tl.load(V_block_ptr, boundary_check=(0,1))

        S_ij = tl.dot(Q_i.to(tl.float32), K_j.to(tl.float32)) * scaler
        if is_causal:
            # Causal masking: query position >= key position
            # Use static ranges and add offsets
            q_indices = tl.arange(0, Q_TILE_SIZE)[:, None] + query_tile_index * Q_TILE_SIZE
            k_indices = tl.arange(0, K_TILE_SIZE)[None, :] + j * K_TILE_SIZE
            mask = q_indices >= k_indices
            S_ij = tl.where(mask, S_ij, float('-inf'))
        
        m_i_new = tl.maximum(m_i, tl.max(S_ij, 1))
        #scaling the score
        P_ij = tl.exp(S_ij - m_i_new[:, None])
        #动态更新logexpsum
        m_scaler = tl.exp(m_i - m_i_new)
        l_i = m_scaler * l_i + tl.sum(P_ij, 1)
        #动态更新O
        o_i = o_i * m_scaler[:, None] + tl.dot(P_ij.to(tl.float32), V_j.to(tl.float32))
        m_i = m_i_new
    
    o_i = o_i * (1.0 / l_i[:, None])
    l_i = tl.log(l_i) + m_i 

    tl.store(O_block_ptr, o_i.to(O_ptr.type.element_ty), boundary_check=(0,1))
    tl.store(L_block_ptr, l_i.to(L_ptr.type.element_ty), boundary_check=(0,))



    
class flash_attention_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        batch_size, N_q, D = q.shape
        _, N_k, _ = k.shape

        # Adaptive tile sizes based on d_model to avoid shared memory overflow
        if D <= 32:
            Q_TILE_SIZE, K_TILE_SIZE = 128, 128
        elif D <= 64:
            Q_TILE_SIZE, K_TILE_SIZE = 64, 64
        else:
            Q_TILE_SIZE, K_TILE_SIZE = 32, 32

        T_q = math.ceil(N_q / Q_TILE_SIZE)
        O_ptr = torch.empty_like(q)
        L_ptr = torch.empty((batch_size, N_q), dtype=torch.float32, device=q.device)

        flash_fwd_kernel[(T_q, batch_size)](
            q, k, v,
            O_ptr, L_ptr,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            O_ptr.stride(0), O_ptr.stride(1), O_ptr.stride(2),
            L_ptr.stride(0), L_ptr.stride(1),
            N_q, N_k,
            D**(-0.5),
            D, Q_TILE_SIZE, K_TILE_SIZE,
            is_causal
        )
        ctx.save_for_backward(q, k, v, O_ptr, L_ptr)
        ctx.is_causal = is_causal
        return O_ptr

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
        scale = d_model**(-0.5)

        # 使用torch.compile优化的backward实现
        @torch.compile
        def flash_backward_impl(Q, K, V, O, L, dO, is_causal):
            # Convert to float32 for computation
            Q = Q.float()
            K = K.float()
            V = V.float()
            O = O.float()
            dO = dO.float()

            # 计算 D = rowsum(O ◦ dO)
            D = torch.sum(O * dO, dim=-1)  # (batch_size, N_q)

            # 重新计算 S 和 P
            S = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (batch_size, N_q, N_k)

            # Causal masking
            if is_causal:
                N_q = Q.shape[1]
                N_k = K.shape[1]
                q_indices = torch.arange(N_q, device=device)[:, None]
                k_indices = torch.arange(N_k, device=device)[None, :]
                mask = q_indices >= k_indices
                S = torch.where(mask, S, torch.tensor(float('-inf'), device=device))

            # P = exp(S - L)
            P = torch.exp(S - L.unsqueeze(-1))  # (batch_size, N_q, N_k)

            # dV = P^T @ dO
            dV = torch.matmul(P.transpose(-2, -1), dO)  # (batch_size, N_k, d_model)

            # dP = dO @ V^T
            dP = torch.matmul(dO, V.transpose(-2, -1))  # (batch_size, N_q, N_k)

            # dS = P ◦ (dP - D) / sqrt(d)
            dS = P * (dP - D.unsqueeze(-1)) * scale  # (batch_size, N_q, N_k)

            # dQ = dS @ K
            dQ = torch.matmul(dS, K)  # (batch_size, N_q, d_model)

            # dK = dS^T @ Q
            dK = torch.matmul(dS.transpose(-2, -1), Q)  # (batch_size, N_k, d_model)

            return dQ, dK, dV

        dQ, dK, dV = flash_backward_impl(q, k, v, O, L, grad_out, ctx.is_causal)

        # Convert back to original dtype
        dQ = dQ.to(q.dtype)
        dK = dK.to(k.dtype)
        dV = dV.to(v.dtype)

        return dQ, dK, dV, None  # None for is_causal
        




            

            



        