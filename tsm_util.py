import torch
import torch.nn.functional as F

def tsm(tensor, duration, version='zero'):
    # tensor [N*T, C, H, W]
    size = tensor.size()
    tensor = tensor.view((-1, duration) + size[1:])
    # tensor [N, T, C, H, W]    
    pre_tensor, post_tensor, peri_tensor = tensor.split([size[1] // 8,
                                                         size[1] // 8,
                                                         3*size[1] // 4], dim=2)     
    if version == 'zero':
        pre_tensor  = F.pad(pre_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:  , ...] #F.pad(pre_tensor,  (0, 0, 0, 0, 0, 0, 1, 0))[:,  :-1, ...]
        post_tensor = F.pad(post_tensor,  (0, 0, 0, 0, 0, 0, 1, 0))[:,  :-1, ...] #F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:  , ...]
    elif version == 'circulant':
        pre_tensor  = torch.cat((pre_tensor [:, -1:  , ...],
                                 pre_tensor [:,   :-1, ...]), dim=1)
        post_tensor = torch.cat((post_tensor[:,  1:  , ...],
                                 post_tensor[:,   :1 , ...]), dim=1)
    else:
        raise ValueError('Unknown TSM version: {}'.format(version))
    return torch.cat((pre_tensor, post_tensor, peri_tensor), dim=2).view(size)