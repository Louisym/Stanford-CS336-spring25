import torch
from torch import nn
# import torch.optim.optimizer
from torch.optim import Optimizer
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
class AdamW(Optimizer):
    def __init__(self,params: Iterable[torch.nn.parameter.Parameter], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8, weight_decay: float = 0.01):
        # if alpha<0:
        #     raise ValueError(f'invalid value input for alpha: {alpha}')
        # if beta[0]<0:
        #     raise ValueError(f'invalid input for beta_1: {beta[0]}')
        # if beta[1]<0:
        #     raise ValueError(f'invalid input for beta_2: {beta[1]}')
        # if eps<0:
        #     raise ValueError(f'invalid input for eps: {eps}')
        # if lamb<0:
        #     raise ValueError(f'invalid input for lambda: {lamb}')
        defaults = {
            'alpha': lr,
            'beta1': betas[0],
            'beta2': betas[1],
            'eps': eps,
            'lamb': weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            closure()
        '''
        param_groups looks like a bunch of (learnable para, hypter_para) pairs:
        [
        {"params": [...layer1 参数...], "lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01},
        {"params": [...layer2 参数...], "lr": 5e-4, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}
        ]
        '''
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                #Embedding layer always uses sparse martix. But AdamW doesn't support it
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")
                alpha = group['alpha']
                beta_1 = group['beta1']
                beta_2 = group['beta2']
                eps = group['eps']
                lamba = group['lamb']
                state = self.state[p]
                prev_m = state.get('m', torch.zeros_like(grad))
                state['m'] = beta_1*prev_m+(1-beta_1)*grad
                prev_v = state.get('v', torch.zeros_like(grad))
                state['v'] = beta_2*prev_v+(1-beta_2)*torch.square(grad)
                t = state.get('t', 1)
                alpha_t = alpha*math.sqrt(1-beta_2**t)/(1-beta_1**t)
                p.data -= alpha_t*state['m']/(torch.sqrt(state['v'])+eps)
                p.data -= alpha*lamba*p.data
                state['t'] = t+1
        return loss

