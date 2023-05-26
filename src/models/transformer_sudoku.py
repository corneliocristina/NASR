from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import Transformer
from models.perception import SequentialPerception

class TransformerSudoku(nn.Module):
    def __init__(self, block_len=256, **kwargs):
        super().__init__()
        self.saved_log_probs = []
        self.rewards = []
        self.perception = SequentialPerception()
        self.nn_solver = Transformer(in_chans=10, num_classes=9,    
                        block_len=block_len, embed_dim=192, depth=4, num_heads=3, mlp_ratio=4,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        self.mask_nn = Transformer(in_chans=9, num_classes=1,
                        block_len=block_len, embed_dim=192, depth=4, num_heads=3, mlp_ratio=4,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        

    def load_pretrained_models(self, dataset):
        perception_path = 'outputs/perception/'+dataset+'/checkpoint_best.pth'
        nn_sol_path = 'outputs/solvernn/'+dataset+'/checkpoint_best.pth'
        mask_nn_path = 'outputs/mask/'+dataset+'/checkpoint_best.pth'

        self.perception.load_state_dict(torch.load(perception_path, map_location='cpu'))
        self.nn_solver.load_state_dict(torch.load(nn_sol_path, map_location='cpu'))
        self.mask_nn.load_state_dict(torch.load(mask_nn_path, map_location='cpu'))

    def forward(self, x, nasr = 'rl'):
        
        if nasr == 'pretrained':
            # for eval of pretrained pipeline (NASR w/o RL)
            assert not bool(self.training), f'{nasr} is available only to evaluate. If you want to train it, use the RL pipeline.'
            x0 = self.perception.forward(x)
            x0 = torch.exp(x0)
            a = x0.argmax(dim=2)
            x1 = F.one_hot(a,num_classes=10).to(torch.float32)
            x2 = self.nn_solver.forward(x1)
            b = x2.argmax(dim=2)+1
            x2 = F.one_hot(b,num_classes=10)
            x2 = x2[:,:,1:].to(torch.float32).to(torch.float32)
            x3 = self.mask_nn.forward(x2)
        else:
            # for traning with RL and eval with RL (NASR with RL)
            assert nasr == 'rl', f'{nasr} do not exists, choose between rl and pretrained'
            x0 = self.perception.forward(x)
            x1 = torch.exp(x0)
            x2 = self.nn_solver.forward(x1)
            x2 = F.softmax(x2, dim=2)
            #x2 = F.gumbel_softmax(x2, tau = 1, hard=True, dim=2)
            x3 = self.mask_nn.forward(x2)
        return x2, x3


def get_model(block_len=256, **kwargs):
    model = TransformerSudoku(block_len=block_len, **kwargs)
    return model

