import torch
import torch.nn.functional as F

# def tsm(tensor, duration, version='zero'):
#     # tensor [N*T, C, H, W]
#     size = tensor.size()
#     tensor = tensor.view((-1, duration) + size[1:])
#     peri_size = (size[1] //3) + (size[1] %3)
#     # tensor [N, T, C, H, W]    
#     pre_tensor, post_tensor, peri_tensor = tensor.split([size[1] // 3,
#                                                          size[1] // 3,
#                                                          peri_size], dim=2)
#     if version == 'zero':
#         pre_tensor  = F.pad(pre_tensor,  (0, 0, 0, 0, 0, 0, 1, 0))[:,  :-1, ...]
#         post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:  , ...]
#     elif version == 'circulant':
#         pre_tensor  = torch.cat((pre_tensor [:, -1:  , ...],
#                                  pre_tensor [:,   :-1, ...]), dim=1)
#         post_tensor = torch.cat((post_tensor[:,  1:  , ...],
#                                  post_tensor[:,   :1 , ...]), dim=1)
#     else:
#         raise ValueError('Unknown TSM version: {}'.format(version))
#     return torch.cat((pre_tensor, post_tensor, peri_tensor), dim=2).view(size)

def tsm_2(tensor, duration, version='zero', remainder =0):
    # tensor [N,C, T, H, W]
    size = tensor.size()
#     tensor = tensor.view((-1, duration) + size[1:])
    tensor = tensor.permute(0,2,1,3,4).contiguous()
    # tensor [N, T, C, H, W]    
    if (remainder ==0):
        pre_tensor, post_tensor, peri_tensor = tensor.split([size[1] // 4,
                                                         size[1] // 4,
                                                         size[1] // 2], dim=2)
    elif (remainder ==1):
        peri_tensor, pre_tensor, post_tensor = tensor.split([size[1] // 2,
                                                         size[1] // 4,
                                                         size[1] // 4], dim=2)
    elif (remainder ==2):
        post_tensor, peri_tensor, pre_tensor = tensor.split([size[1] // 4,
                                                         size[1] // 2,
                                                         size[1] // 4], dim=2)        
    if version == 'zero':
        pre_tensor  = F.pad(pre_tensor,  (0, 0, 0, 0, 0, 0, 1, 0))[:,  :-1, ...]
        post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:  , ...]
    elif version == 'circulant':
        pre_tensor  = torch.cat((pre_tensor [:, -1:  , ...],
                                 pre_tensor [:,   :-1, ...]), dim=1)
        post_tensor = torch.cat((post_tensor[:,  1:  , ...],
                                 post_tensor[:,   :1 , ...]), dim=1)
    else:
        raise ValueError('Unknown TSM version: {}'.format(version))
    return torch.cat((pre_tensor, post_tensor, peri_tensor), dim=2).view(size)


def tsm(tensor, duration, version='zero', remainder =0):
    # tensor [N*T, C, H, W]
    size = tensor.size()
    tensor = tensor.view((-1, duration) + size[1:])
    # tensor [N, T, C, H, W]    
    if (remainder ==0):
        pre_tensor, post_tensor, peri_tensor = tensor.split([size[1] // 8,
                                                         size[1] // 8,
                                                         3*size[1] // 4], dim=2)
    elif (remainder ==1):
        peri_tensor, pre_tensor, post_tensor = tensor.split([3*size[1] // 4,
                                                         size[1] // 8,
                                                         size[1] // 8], dim=2)
    elif (remainder ==2):
        post_tensor, peri_tensor, pre_tensor = tensor.split([size[1] // 8,
                                                         3*size[1] // 4,
                                                         size[1] // 8], dim=2)        
    if version == 'zero':
        pre_tensor  = F.pad(pre_tensor,  (0, 0, 0, 0, 0, 0, 1, 0))[:,  :-1, ...]
        post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:  , ...]
    elif version == 'circulant':
        pre_tensor  = torch.cat((pre_tensor [:, -1:  , ...],
                                 pre_tensor [:,   :-1, ...]), dim=1)
        post_tensor = torch.cat((post_tensor[:,  1:  , ...],
                                 post_tensor[:,   :1 , ...]), dim=1)
    else:
        raise ValueError('Unknown TSM version: {}'.format(version))
    return torch.cat((pre_tensor, post_tensor, peri_tensor), dim=2).view(size)

def tsm_arrange(tensor, duration, version='zero'):
    # tensor [N*T, C, H, W]
    size = tensor.size()
    tensor = tensor.view((-1, duration) + size[1:])
    # tensor [N, T, C, H, W]
    peri_tensor, pre_tensor, post_tensor= tensor.split([size[1] // 2,
                                                         size[1] // 4,
                                                         size[1] // 4], dim=2)
    if version == 'zero':
        pre_tensor  = F.pad(pre_tensor,  (0, 0, 0, 0, 0, 0, 1, 0))[:,  :-1, ...]
        post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:  , ...]
    elif version == 'circulant':
        pre_tensor  = torch.cat((pre_tensor [:, -1:  , ...],
                                 pre_tensor [:,   :-1, ...]), dim=1)
        post_tensor = torch.cat((post_tensor[:,  1:  , ...],
                                 post_tensor[:,   :1 , ...]), dim=1)
    else:
        raise ValueError('Unknown TSM version: {}'.format(version))
    return torch.cat((peri_tensor, pre_tensor, post_tensor), dim=2).view(size)

def tsm_pre(tensor, duration, version='zero'):
    # tensor [N*T, C, H, W]
    size = tensor.size()
    tensor = tensor.view((-1, duration) + size[1:])
    # tensor [N, T, C, H, W]
    pre_tensor, peri_tensor = tensor.split([size[1] // 2,
                                                         size[1] // 2], dim=2)
    if version == 'zero':
        pre_tensor  = F.pad(pre_tensor,  (0, 0, 0, 0, 0, 0, 1, 0))[:,  :-1, ...]
    elif version == 'circulant':
        pre_tensor  = torch.cat((pre_tensor [:, -1:  , ...],
                                 pre_tensor [:,   :-1, ...]), dim=1)
    else:
        raise ValueError('Unknown TSM version: {}'.format(version))
    return torch.cat((pre_tensor, peri_tensor), dim=2).view(size)

def tsm_post(tensor, duration, version='zero'):
    # tensor [N*T, C, H, W]
    size = tensor.size()
    tensor = tensor.view((-1, duration) + size[1:])
    # tensor [N, T, C, H, W]
    post_tensor, peri_tensor = tensor.split([size[1] // 2,
                                                         size[1] // 2], dim=2)
    if version == 'zero':
        post_tensor  = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:  , ...]
    elif version == 'circulant':
        post_tensor  = torch.cat((post_tensor [:, -1:  , ...],
                                 post_tensor [:,   :-1, ...]), dim=1)
    else:
        raise ValueError('Unknown TSM version: {}'.format(version))
    return torch.cat((post_tensor, peri_tensor), dim=2).view(size)

def roll(x, n):
    return torch.cat((x[:,-n:, ...], x[:,:-n, ...]), dim=1)

def tsm_longrange(tensor, duration, version='zero'):
    # tensor [N*T, C, H, W]
    size = tensor.size()
    tensor = tensor.view((-1, duration) + size[1:])
    # tensor [N, T, C, H, W]
    roll_0, roll_1, roll_2, roll_3, roll_4, roll_5, roll_6, roll_7 = tensor.split([size[1] // 8, size[1] //8, size[1] // 8, size[1] //8, 
                                                                                  size[1] // 8, size[1] //8, size[1] // 8, size[1] //8], dim=2)
    if version == 'zero':
        roll_1 = roll(roll_1,1)
        roll_2 = roll(roll_2,2)
        roll_3 = roll(roll_3,3)
        roll_4 = roll(roll_4,4)
        roll_5 = roll(roll_5,5)
        roll_6 = roll(roll_6,6)
        roll_7 = roll(roll_7,7)       
    else:
        raise ValueError('Unknown TSM version: {}'.format(version))
    return torch.cat((roll_0, roll_1, roll_2, roll_3, roll_4, roll_5, roll_6, roll_7), dim=2).view(size)      

def roll_2(x, n):
    return torch.cat((x[:, :,-n:], x[:,:,:-n]), dim=2)

def tsm_extend(tensor, duration, version='zero'):
    # tensor [N*T, C, H, W]
    size = tensor.size()
    tensor = tensor.view((-1, duration) + size[1:])
    # tensor [N, T, C, H, W]
    pre_tensor, post_tensor, peri_tensor = tensor.split([size[1] // 4,
                                                         size[1] // 4,
                                                         size[1] // 2], dim=2)
    if version == 'zero':
        pre_tensor  = F.pad(pre_tensor,  (0, 0, 0, 0, 0, 0, 1, 0))[:,  :-1, ...]
        pre_size = pre_tensor.size()
        pre_c_size = (pre_size[2] // 5) + (pre_size[2] % 5)
        pre_c, pre_r, pre_l, pre_d, pre_u = pre_tensor.split( [pre_c_size, pre_size[2] //5, pre_size[2]// 5, pre_size[2]//5, pre_size[2]//5 ], dim=2)
        pre_r = F.pad(pre_r, (1,0))[:,:,:,:,:-1]
        pre_l = F.pad(pre_l, (0,1))[:,:,:,:,1:]
        pre_d = F.pad(pre_d, (0,0,1,0))[:,:,:,:-1,:]
        pre_u = F.pad(pre_u, (0,0,0,1))[:,:,:,1:,:]
        pre_tensor = torch.cat((pre_c,pre_r,pre_l, pre_d, pre_u), dim=2)
        
        post_size = post_tensor.size()
        post_c_size = (post_size[2] // 5) + (post_size[2] % 5)        
        post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:  , ...]
        post_c, post_r, post_l, post_d, post_u = post_tensor.split( [post_c_size, post_size[2] //5, post_size[2]//5, post_size[2]//5, post_size[2]//5], dim=2)
        post_r = F.pad(post_r, (1,0))[:,:,:,:,:-1]
        post_l = F.pad(post_l, (0,1))[:,:,:,:,1:]
        post_d = F.pad(post_d, (0,0,1,0))[:,:,:,:-1,:]
        post_u = F.pad(post_u, (0,0,0,1))[:,:,:,1:,:] 
        post_tensor = torch.cat((post_c, post_r,post_l,post_d,post_u), dim=2)
        
    elif version == 'circulant':
        pre_tensor  = torch.cat((pre_tensor [:, -1:  , ...],
                                 pre_tensor [:,   :-1, ...]), dim=1)
        post_tensor = torch.cat((post_tensor[:,  1:  , ...],
                                 post_tensor[:,   :1 , ...]), dim=1)
    else:
        raise ValueError('Unknown TSM version: {}'.format(version))
    return torch.cat((pre_tensor, post_tensor, peri_tensor), dim=2).view(size)        

def tsm_extend_v2(tensor, duration, version='zero'):
    # tensor [N*T, C, H, W]
    size = tensor.size()
    tensor = tensor.view((-1, duration) + size[1:])
    # tensor [N, T, C, H, W]
    pre_tensor, post_tensor, peri_tensor = tensor.split([size[1] // 4,
                                                         size[1] // 4,
                                                         size[1] // 2], dim=2)
    if version == 'zero':
        pre_tensor  = F.pad(pre_tensor,  (0, 0, 0, 0, 0, 0, 1, 0))[:,  :-1, ...]
        pre_size = pre_tensor.size()        
        pre_r, pre_l, pre_d, pre_u = pre_tensor.split( [pre_size[2] //4, pre_size[2]// 4, pre_size[2]//4, pre_size[2]//4 ], dim=2)
        pre_r = F.pad(pre_r, (1,0))[:,:,:,:,:-1]
        pre_l = F.pad(pre_l, (0,1))[:,:,:,:,1:]
        pre_d = F.pad(pre_d, (0,0,1,0))[:,:,:,:-1,:]
        pre_u = F.pad(pre_u, (0,0,0,1))[:,:,:,1:,:]
        pre_tensor_ex = torch.cat((pre_r,pre_l, pre_d, pre_u), dim=2)
        
        post_size = post_tensor.size()       
        post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:  , ...]
        post_r, post_l, post_d, post_u = post_tensor.split( [post_size[2] //4, post_size[2]//4, post_size[2]//4, post_size[2]//4], dim=2)
        post_r = F.pad(post_r, (1,0))[:,:,:,:,:-1]
        post_l = F.pad(post_l, (0,1))[:,:,:,:,1:]
        post_d = F.pad(post_d, (0,0,1,0))[:,:,:,:-1,:]
        post_u = F.pad(post_u, (0,0,0,1))[:,:,:,1:,:] 
        post_tensor_ex = torch.cat((post_r,post_l,post_d,post_u), dim=2)
        
    elif version == 'circulant':
        pre_tensor  = torch.cat((pre_tensor [:, -1:  , ...],
                                 pre_tensor [:,   :-1, ...]), dim=1)
        post_tensor = torch.cat((post_tensor[:,  1:  , ...],
                                 post_tensor[:,   :1 , ...]), dim=1)
    else:
        raise ValueError('Unknown TSM version: {}'.format(version))
    return torch.cat((pre_tensor_ex, post_tensor_ex, peri_tensor), dim=2).view(size)

def tsm_grad(tensor, duration, version='zero'):
    # tensor [N*T, C, H, W]
    size = tensor.size()
    tensor = tensor.view((-1, duration) + size[1:])
    # tensor [N, T, C, H, W]
    tensor_pad = F.pad(tensor, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1  , ...]
# print (tensor_pad[0,0,...])
    pre_size = tensor_pad.size()
    pre_c_size = (pre_size[2] // 5) + (pre_size[2] % 5)
    tensor_c, tensor_r, tensor_l, tensor_d, tensor_u = tensor.split( [pre_c_size, pre_size[2] //5, pre_size[2]//5, pre_size[2]//5, pre_size[2]//5 ], dim=2)
    pre_c, pre_r, pre_l, pre_d, pre_u = tensor_pad.split( [pre_c_size, pre_size[2] //5, pre_size[2]//5, pre_size[2]//5, pre_size[2]//5 ], dim=2)
    pre_r = F.pad(pre_r, (1,0))[:,:,:,:,:-1]
    pre_l = F.pad(pre_l, (0,1))[:,:,:,:,1:]
    pre_d = F.pad(pre_d, (0,0,1,0))[:,:,:,:-1,:]
    pre_u = F.pad(pre_u, (0,0,0,1))[:,:,:,1:,:]

    pre_r[:,:,:,:,0]=tensor_r[:,:,:,:,0]
    pre_l[:,:,:,:,-1]=tensor_l[:,:,:,:,-1]
    pre_d[:,:,:,0,:]=tensor_d[:,:,:,0,:]
    pre_u[:,:,:,-1,:]=tensor_u[:,:,:,-1,:]

    tensor_pad = torch.cat((pre_c,pre_r,pre_l, pre_d, pre_u), dim=2)
    tensor_pad[:,0,...] = tensor[:,0,...]
    grad = tensor- tensor_pad

    return grad.view(size)

def tsm_grad_v2(tensor, duration, version='zero'):
    # tensor [N*T, C, H, W]
    size = tensor.size()
    tensor = tensor.view((-1, duration) + size[1:])
    # tensor [N, T, C, H, W]
    tensor_pad = F.pad(tensor, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1  , ...]
    tensor_pad[:,0,...] = tensor[:,0,...]
    grad = tensor- tensor_pad

    return grad.view(size)

def tsm_grad_v3(tensor, duration, version='zero'):
    # tensor [N*T, C, H, W]
    size = tensor.size()
    tensor = tensor.view((-1, duration) + size[1:])
    # tensor [N, T, C, H, W]
    tensor_pad = F.pad(tensor, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1  , ...]
    tensor_pad[:,0,...] = tensor[:,0,...]
    grad = tensor- tensor_pad
    grad_pos = grad* (torch.sign(grad)+1) /2
    grad_neg = grad* (torch.sign(grad) -1) /2
    
    return torch.cat((grad_pos, grad_neg), dim=2).view((size[0], 2*size[1],size[2],size[3]))


# def tsm_extend(tensor, duration, version='zero'):
#     # tensor [N*T, C, H, W]
#     size = tensor.size()
#     tensor = tensor.view((-1, duration) + size[1:])
#     # tensor [N, T, C, H, W]
#     pre_tensor, post_tensor, peri_tensor = tensor.split([size[1] // 4,
#                                                          size[1] // 4,
#                                                          size[1] // 2], dim=2)
#     if version == 'zero':
#         pre_tensor  = F.pad(pre_tensor,  (0, 0, 0, 0, 0, 0, 1, 0))[:,  :-1, ...]
#         pre_size = pre_tensor.size()
#         pre_c_size =  (pre_size[2] // 9) + (pre_size[2] % 9)
#         pre_c, pre_r, pre_l, pre_d, pre_u, pre_dr, pre_dl, pre_ur, pre_ul = pre_tensor.split( [pre_c_size, pre_size[2] //9, pre_size[2]//9, pre_size[2]//9, pre_size[2]//9, pre_size[2] //9, pre_size[2]//9, pre_size[2]//9, pre_size[2]//9  ], dim=2)
#         pre_r = F.pad(pre_r, (1,0))[:,:,:,:,:-1]
#         pre_l = F.pad(pre_l, (0,1))[:,:,:,:,1:]
#         pre_d = F.pad(pre_d, (0,0,1,0))[:,:,:,:-1,:]
#         pre_u = F.pad(pre_u, (0,0,0,1))[:,:,:,1:,:]
#         pre_dr = F.pad(pre_d, (1,0,1,0))[:,:,:,:-1,:-1]
#         pre_dl =  F.pad(pre_d, (0,1,1,0))[:,:,:,:-1,1:]
#         pre_ur = F.pad(pre_u, (1,0,0,1))[:,:,:,1:,:-1]
#         pre_ul = F.pad(pre_u, (0,1,0,1))[:,:,:,1:,1:]
#         pre_tensor = torch.cat((pre_c,pre_r,pre_l, pre_d, pre_u, pre_dr, pre_dl, pre_ur, pre_ul), dim=2)
        
#         post_size = post_tensor.size()
#         post_c_size = (post_size[2] // 9) + (post_size[2] %9)         
#         post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:  , ...]
#         post_c, post_r, post_l, post_d, post_u, post_dr, post_dl, post_ur, post_ul = post_tensor.split( [post_c_size , post_size[2] //9, post_size[2]//9, post_size[2]//9, post_size[2]//9, post_size[2] //9, post_size[2]//9, post_size[2]//9, post_size[2]//9  ], dim=2)
#         post_r = F.pad(post_r, (1,0))[:,:,:,:,:-1]
#         post_l = F.pad(post_l, (0,1))[:,:,:,:,1:]
#         post_d = F.pad(post_d, (0,0,1,0))[:,:,:,:-1,:]
#         post_u = F.pad(post_u, (0,0,0,1))[:,:,:,1:,:]
#         post_dr = F.pad(post_d, (1,0,1,0))[:,:,:,:-1,:-1]
#         post_dl =  F.pad(post_d, (0,1,1,0))[:,:,:,:-1,1:]
#         post_ur = F.pad(post_u, (1,0,0,1))[:,:,:,1:,:-1]
#         post_ul = F.pad(post_u, (0,1,0,1))[:,:,:,1:,1:]
#         post_tensor = torch.cat((post_c, post_r, post_l, post_d, post_u, post_dr, post_dl, post_ur, post_ul), dim=2)
        
#     elif version == 'circulant':
#         pre_tensor  = torch.cat((pre_tensor [:, -1:  , ...],
#                                  pre_tensor [:,   :-1, ...]), dim=1)
#         post_tensor = torch.cat((post_tensor[:,  1:  , ...],
#                                  post_tensor[:,   :1 , ...]), dim=1)
#     else:
#         raise ValueError('Unknown TSM version: {}'.format(version))
#     return torch.cat((pre_tensor, post_tensor, peri_tensor), dim=2).view(size)

# def tsm_extend_v2(tensor, duration, version='zero'):
#     # tensor [N*T, C, H, W]
#     size = tensor.size()
#     tensor = tensor.view((-1, duration) + size[1:])
#     # tensor [N, T, C, H, W]
#     pre_tensor, post_tensor, peri_tensor = tensor.split([size[1] // 4,
#                                                          size[1] // 4,
#                                                          size[1] // 2], dim=2)
#     if version == 'zero':
#         pre_tensor  = F.pad(pre_tensor,  (0, 0, 0, 0, 0, 0, 1, 0))[:,  :-1, ...]
#         pre_size = pre_tensor.size()
#         pre_c, pre_r, pre_l, pre_d, pre_u, pre_dr, pre_dl, pre_ur, pre_ul = pre_tensor.split( [pre_size[2] // 2, pre_size[2] //16, pre_size[2]//16, pre_size[2]//16, pre_size[2]//16, pre_size[2] //16, pre_size[2]//16, pre_size[2]//16, pre_size[2]//16  ], dim=2)
#         pre_r = F.pad(pre_r, (1,0))[:,:,:,:,:-1]
#         pre_l = F.pad(pre_l, (0,1))[:,:,:,:,1:]
#         pre_d = F.pad(pre_d, (0,0,1,0))[:,:,:,:-1,:]
#         pre_u = F.pad(pre_u, (0,0,0,1))[:,:,:,1:,:]
#         pre_dr = F.pad(pre_d, (1,0,1,0))[:,:,:,:-1,:-1]
#         pre_dl =  F.pad(pre_d, (0,1,1,0))[:,:,:,:-1,1:]
#         pre_ur = F.pad(pre_u, (1,0,0,1))[:,:,:,1:,:-1]
#         pre_ul = F.pad(pre_u, (0,1,0,1))[:,:,:,1:,1:]
#         pre_tensor = torch.cat((pre_c,pre_r,pre_l, pre_d, pre_u, pre_dr, pre_dl, pre_ur, pre_ul), dim=2)
        
#         post_size = post_tensor.size()
#         post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:  , ...]
#         post_c, post_r, post_l, post_d, post_u, post_dr, post_dl, post_ur, post_ul = post_tensor.split( [post_size[2] // 2, post_size[2] //16, post_size[2]//16, post_size[2]//16, post_size[2]//16, post_size[2] //16, post_size[2]//16, post_size[2]//16, post_size[2]//16  ], dim=2)
#         post_r = F.pad(post_r, (1,0))[:,:,:,:,:-1]
#         post_l = F.pad(post_l, (0,1))[:,:,:,:,1:]
#         post_d = F.pad(post_d, (0,0,1,0))[:,:,:,:-1,:]
#         post_u = F.pad(post_u, (0,0,0,1))[:,:,:,1:,:]
#         post_dr = F.pad(post_d, (1,0,1,0))[:,:,:,:-1,:-1]
#         post_dl =  F.pad(post_d, (0,1,1,0))[:,:,:,:-1,1:]
#         post_ur = F.pad(post_u, (1,0,0,1))[:,:,:,1:,:-1]
#         post_ul = F.pad(post_u, (0,1,0,1))[:,:,:,1:,1:]
#         post_tensor = torch.cat((post_c, post_r, post_l, post_d, post_u, post_dr, post_dl, post_ur, post_ul), dim=2)
        
#     elif version == 'circulant':
#         pre_tensor  = torch.cat((pre_tensor [:, -1:  , ...],
#                                  pre_tensor [:,   :-1, ...]), dim=1)
#         post_tensor = torch.cat((post_tensor[:,  1:  , ...],
#                                  post_tensor[:,   :1 , ...]), dim=1)
#     else:
#         raise ValueError('Unknown TSM version: {}'.format(version))
#     return torch.cat((pre_tensor, post_tensor, peri_tensor), dim=2).view(size)
