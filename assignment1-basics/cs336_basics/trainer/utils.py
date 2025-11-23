import torch
from torch import nn
import math
from collections.abc import Iterable
def cross_entropy(out_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    '''
    out_logit: torch.Tensor Output of the Linear of the transformer. Shape is like (batch_size, vocab_size)
    target: torch.Tensor We are using next-word prediction, so the target is the next word of our input. Shape is like (batch_size)
    '''
    get_logit = out_logit.gather(dim=-1, index=target.unsqueeze(-1))#得到对应id的token的概率
    logsumexp = torch.logsumexp(input=out_logit, dim=-1, keepdim=True)#得到logsumexp
    loss = -get_logit + logsumexp#这就是把负对数似然和化简后的结果(batch_size, 1)
    #don't forget we need the average loss in case of overflow
    return torch.mean(loss,dim=0, keepdim=False)
    

def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    assert cosine_cycle_iters>warmup_iters, 'Invalid input for iteration striction'
    if it<warmup_iters:
        return it*max_learning_rate/warmup_iters
    elif warmup_iters<=it<=cosine_cycle_iters:
        return min_learning_rate+0.5*(1+math.cos((it-warmup_iters)*math.pi/(cosine_cycle_iters-warmup_iters)))*(max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    grads = [p.grad for p in parameters if p.grad is not None]
    L2_norm = 0.0
    for g in grads:
        L2_norm += (g.data**2).sum()
    L2_norm = torch.sqrt(L2_norm)
    if L2_norm < max_l2_norm:
        pass
    else:
        for g in grads:
            g.data *= max_l2_norm/(L2_norm+eps)
    
