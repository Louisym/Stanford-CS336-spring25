import torch
import torch.nn
def save_checkpoint(model, optimizer, iteration, out):
    '''
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    iteration: int
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    '''
    
    torch.save(
        obj={
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'iteration': iteration
        },
        f=out
    )
def load_checkpoint(src, model, optimizer) -> torch.long:
    '''
    should load a checkpoint from src (path or file-
like object), and then recover the model and optimizer states from that checkpoint. Your
function should return the iteration number that was saved to the checkpoint. You can use
torch.load(src) to recover what you saved in your save_checkpoint implementation, and the
load_state_dict method in both the model and optimizers to return them to their previous
states
    
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    '''
    ckp = torch.load(src, map_location='cpu')
    model.load_state_dict(ckp['model_state'])
    optimizer.load_state_dict(ckp['optimizer_state'])
    return ckp['iteration']
