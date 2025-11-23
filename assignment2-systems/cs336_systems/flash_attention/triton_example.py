from torch import device
import triton
import triton.language as tl
import einops
import torch
@triton.jit
def weighted_sum_forward(
        x_ptr, w_ptr, out_ptr,
        x_stride_row, x_stride_dim,
        weight_stride_dim,
        output_stride_row,
        ROWS, D,
        ROWS_TILE_SIZE, D_TILE_SIZE
):
    row_tile_idx = tl.program_id(0)
    x_block_ptr = tl.make_block_ptr(x_ptr, shape=(ROWS, D,), strides=(x_stride_row, x_stride_dim), offsets=(row_tile_idx*ROWS_TILE_SIZE, 0), block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE), order=(1, 0))
    weight_block_ptr = tl.make_block_ptr(w_ptr, shape=(D,), strides=(weight_stride_dim, ), offsets=(0, ), block_shape=(D_TILE_SIZE,), order=(0,))
    output_block_ptr = tl.make_block_ptr(out_ptr, shape=(ROWS, ), strides=(output_stride_row, ), offsets=(ROWS_TILE_SIZE*row_tile_idx,), block_shape=(ROWS_TILE_SIZE,), order=(0,))
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero") # (D_TILE_SIZE,)
        output += tl.sum(row*weight[None, :], axis=1)

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

    tl.store(output_block_ptr, output, boundary_check=(0,))

class WeightedSumFunc(torch.autograd.Function):
@staticmethod
    def forward(ctx, x, weight):
        D, out_dim = x.shape[-1], x.shape[:-1]
        x = einops.rearrange(x, '... d -> (...) d')
        ctx.save_for_backward(x, weight)
        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"
        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16
        ctx.ROWS_TILE_SIZE =  16
        ctx.input_shape = x.shape

        y = torch.empty(out_dim, device=device)
        n_rows = y.numel()
        weighted_sum_forward[triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE)](
            x, weight, y,
            x_stride_row=x.stride(0), x_stride_dim=x.stride(1),
            weight_stride_dim=weight.stride(0), output_stride_row=y.stride(0),
            ROWS=n_rows, D=D, 
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        return y.view(out_dim)

@triton.jit
def weighted_sum_backward(
 x_ptr, weight_ptr, # Input
 grad_output_ptr, # Grad input
 grad_x_ptr, partial_grad_weight_ptr, # Grad outputs
 stride_xr, stride_xd,
 stride_wd,
 stride_gr,
 stride_gxr, stride_gxd,
 stride_gwb, stride_gwd,
 NUM_ROWS, D,
 ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
 ):
    row_tile_idx = tl.program_id(0)
    num_rows = tl.num_programs(0)
    # grad_output: [NUM_ROWS]
    grad_output_block_ptr = tl.make_block_ptr(
        base=grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # x: [NUM_ROWS, D]
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # weight: [D]
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    # grad_x: [NUM_ROWS, D]
    grad_x_block_ptr = tl.make_block_ptr(
        base=grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # partial_grad_weight: [n_row_tiles, D]
    # （每个 row tile 写一行部分和，之后在 kernel 外 sum(dim=0) 合并）
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        base=partial_grad_weight_ptr,
        shape=(n_row_tiles, D),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )
    