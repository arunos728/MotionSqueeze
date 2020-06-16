"""
An example combining `Temporal Shift Module` with `ResNet`. This implementation
is based on `Temporal Segment Networks`, which merges temporal dimension into
batch, i.e. inputs [N*T, C, H, W]. Here we show the case with residual connections
and zero padding with 8 frames as input.
"""
import torch.nn as nn
import torch.nn.functional as F
import torch as tr
from tsm_util import tsm
import torch.utils.model_zoo as model_zoo
from spatial_correlation_sampler import SpatialCorrelationSampler

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

class NonLocal(nn.Module):
    def __init__(self, in_channel=256):
        super(NonLocal, self).__init__()
        self.theta = nn.Conv3d(in_channel, in_channel //2 , kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv3d(in_channel, in_channel //2, kernel_size=1, stride=1, padding=0)
        self.g = nn.Conv3d(in_channel, in_channel //2, kernel_size=1, stride=1, padding=0)
        self.embed = nn.Conv3d(in_channel //2, in_channel, kernel_size=1, stride=1, padding=0) 

    def forward(self, feat):
        size = feat.size()      
        feat2 = feat.view((-1, 8)+ size[1:] )   # B T C H W         
        feat2 = feat2.permute(0,2,1,3,4).contiguous() # B C T H W
        b, c, t, h, w = feat2.size()
        theta_ = self.theta(feat2).view(b, c//2, t*h * w)
        phi_ = self.phi(feat2).view(b, c//2, t*h * w).permute(0, 2, 1).contiguous()
        g_ = self.g(feat2).view(b, c//2, t*h * w)

        attention = F.softmax(tr.matmul(phi_, theta_), dim=1)

        new_feat = self.embed(tr.matmul(g_, attention).view(b, c//2,t, h, w))
        new_feat = new_feat.permute(0,2,1,3,4).contiguous().view(b*t, c,h,w)
        return feat + new_feat    
    
class BasicBlock_NL(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_NL, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        # NL
        self.theta = nn.Conv3d(planes, planes //2 , kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv3d(planes, planes //2, kernel_size=1, stride=1, padding=0)
        self.g = nn.Conv3d(planes, planes //2, kernel_size=1, stride=1, padding=0)
        self.embed = nn.Conv3d(planes //2, planes, kernel_size=1, stride=1, padding=0)
        self.bn3d = nn.BatchNorm3d(planes)
        
    def forward(self, x):
        identity = x
        out = tsm(x, 8, 'zero')
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)         
            
        out += identity
        out = self.relu(out)

        size = out.size()
        feat2 = out.view((-1, 8)+ size[1:] )   # B T C H W     
#         print (feat2.size())
        feat2 = feat2.permute(0,2,1,3,4).contiguous() # B C T H W
        b, c, t, h, w = feat2.size()
        theta_ = self.theta(feat2).view(b, c//2, t*h * w)
        phi_ = self.phi(feat2).view(b, c//2, t*h * w).permute(0, 2, 1).contiguous()
        g_ = self.g(feat2).view(b, c//2, t*h * w)

        attention = F.softmax(tr.matmul(phi_, theta_), dim=1)
        new_feat = self.embed(tr.matmul(g_, attention).view(b, c//2,t, h, w))
        new_feat = self.bn3d(new_feat)
        new_feat = new_feat.permute(0,2,1,3,4).contiguous().view(b*t, c,h,w)                
        
        return out + new_feat   

class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, num_segments,stride=1, downsample=None, remainder=0):
        super(BasicBlock2, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.remainder =remainder
        self.num_segments = num_segments        

    def forward(self, x):
        identity = x  
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out        
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, num_segments, stride=1, downsample=None, remainder=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.remainder= remainder
        self.num_segments = num_segments

    def forward(self, x):
        identity = x  
        out = tsm(x, self.num_segments, 'zero')   
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
       
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,num_segments, stride=1, downsample=None, remainder=0):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.remainder= remainder        
        self.num_segments = num_segments        

    def forward(self, x):
        identity = x
        out = tsm(x, self.num_segments, 'zero')
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    


class matching_layer(nn.Module):
    def __init__(self, ks, patch, stride, pad, patch_dilation):
        super(matching_layer, self).__init__()
        self.relu = nn.ReLU()
        self.patch = patch
        self.correlation_sampler = SpatialCorrelationSampler(ks, patch, stride, pad, patch_dilation)
        
    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, feature1, feature2):
        feature1 = self.L2normalize(feature1)
        feature2 = self.L2normalize(feature2)
        b, c, h1, w1 = feature1.size()
        b, c, h2, w2 = feature2.size()
        corr = self.correlation_sampler(feature1, feature2)
        corr = corr.view(b, self.patch * self.patch, h1* w1) # Channel : target // Spatial grid : source
        corr = self.relu(corr)
        return corr
    
class flow_refinement(nn.Module):
#     expansion = 4    
    def __init__(self, num_segments, expansion = 1, pos=2, channels=3):
        super(flow_refinement, self).__init__()
        self.num_segments = num_segments
        self.expansion = expansion
        self.channels = channels
        self.pos = pos
        self.bn_a = nn.BatchNorm2d(2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.channels, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )       
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(64)
        )
        self.conv_tsm = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )          
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128*self.expansion, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128*self.expansion)
        )    
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(64, 256*self.expansion, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(256*self.expansion)
        )            
        self.relu = nn.ReLU(inplace=True)    

        
    def forward(self, x, res, match_v):
        _, c1, h1, w1 = res.size()
        x = tr.cat([x, match_v], dim=1)
        _, c, h, w = x.size()       
        _, c2, h, w = match_v.size()
        x = x.view(-1,self.num_segments-1,c,h,w) ## (b,t-1, 2, h,w) .permute(0,2,1,3,4).contiguous()  ## (b,2,t-1,h,w)                        
        size = x.size()
        zero_pad = tr.zeros(size[0],1,c,h,w).to('cuda')          
        x = tr.cat([x,zero_pad], dim=1)   ## (b,t,2,h,w)
        x = x.view(-1,c,h,w)
#         x = x.permute(0,2,1,3,4).contiguous()        
        
            
        
        x = self.conv1(x)
        if (self.pos==1):
            x = self.conv2_2(x)
        else:
#             x_2 = self.conv2_temp(x)            
            x = self.conv2(x)
        if (self.pos==2):
#             x = tsm(x, self.num_segments, 'zero')
#             x = self.conv_tsm(x)
            x = self.conv3(x)
#             x_2 = self.conv3_temp(x_2)
        elif(self.pos==3):
            x = self.conv3_2(x)

        x = x + res
        x = self.relu(x)
        return x  
    
class ResNet(nn.Module):

    def __init__(self, block, block2, layers, non_local_block, num_segments, flow_estimation, non_local=0, num_classes=1000, zero_init_residual=False,
                n_iter=10, learnable=[0,1,1,1,1]):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()          
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax = nn.Softmax(dim=1)        
        self.num_segments = num_segments     
        self.flow_estimation = flow_estimation
                                                                   
      
        ## MatchFlow
        self.patch= 15
        self.patch_dilation =1
        self.matching_layer = matching_layer(ks=1, patch=self.patch, stride=1, pad=0, patch_dilation=self.patch_dilation)       
#         self.matching_layer_l1 = matching_layer(ks=1, patch=self.patch, stride=1, pad=0, patch_dilation=self.patch_dilation)
#         self.matching_layer_l3 = matching_layer(ks=1, patch=self.patch, stride=1, pad=0, patch_dilation=self.patch_dilation)        
                
        self.flow_refinement = flow_refinement(num_segments=num_segments, expansion=block.expansion, channels=3)       
        self.soft_argmax = nn.Softmax(dim=1)
        self.gumbel_softmax = nn.Softmax(dim=1)
             
        
        if (non_local == 0):
            self.layer1 = self._make_layer(block, 64, layers[0], num_segments=num_segments)
            self.layer2 = self._make_layer(block, 128, layers[1],  num_segments=num_segments, stride=2)
            if (self.flow_estimation == 1):
                self.layer3 = self._make_layer(block2, 256, layers[2],  num_segments=num_segments, stride=2)
                self.layer4 = self._make_layer(block2, 512, layers[3],  num_segments=num_segments, stride=2)
            else:                
                self.layer3 = self._make_layer(block2, 256, layers[2],  num_segments=num_segments, stride=2)
                self.layer4 = self._make_layer(block2, 512, layers[3],  num_segments=num_segments, stride=2)                
        else:
            self.layer1 = self._make_layer(block, 64, layers[0], num_segments=num_segments)
            self.layer2 = self._make_layer(block, 128, layers[1], num_segments=num_segments, stride=2)
#             self.layer2 = self._make_non_local_layer(non_local_block, block, 128, layers[1], non_local-1,stride=2)            
            self.layer3 = self._make_non_local_layer(non_local_block, block, 256, layers[2], non_local, num_segments=num_segments,stride=2)
#             self.layer3 = self._make_layer(block2, 256, layers[2], stride=2)        
            self.layer4 = self._make_layer(block, 512, layers[3], num_segments=num_segments, stride=2)            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)                   
        
        
        for m in self.modules():       
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm) 

    
    def apply_binary_kernel(self, match, h, w, region):
        # binary kernel
        x_line = tr.arange(w, dtype=tr.float).to('cuda').detach()
        y_line = tr.arange(h, dtype=tr.float).to('cuda').detach()
        x_kernel_1 = x_line.view(1,1,1,1,w).expand(1,1,w,h,w).to('cuda').detach()
        y_kernel_1 = y_line.view(1,1,1,h,1).expand(1,h,1,h,w).to('cuda').detach()
        x_kernel_2 = x_line.view(1,1,w,1,1).expand(1,1,w,h,w).to('cuda').detach()
        y_kernel_2 = y_line.view(1,h,1,1,1).expand(1,h,1,h,w).to('cuda').detach()

        ones = tr.ones(1).to('cuda').detach()
        zeros = tr.zeros(1).to('cuda').detach()

        eps = 1e-6
        kx = tr.where(tr.abs(x_kernel_1 - x_kernel_2)<=region, ones, zeros).to('cuda').detach()
        ky = tr.where(tr.abs(y_kernel_1 - y_kernel_2)<=region, ones, zeros).to('cuda').detach()
        kernel = kx * ky + eps
        kernel = kernel.view(1,h*w,h*w).to('cuda').detach()                
        return match* kernel


    def apply_gaussian_kernel(self, corr, h,w,p, sigma=5):
        b, c, s = corr.size()

        x = tr.arange(p, dtype=tr.float).to('cuda').detach()
        y = tr.arange(p, dtype=tr.float).to('cuda').detach()

        idx = corr.max(dim=1)[1] # b x hw    get maximum value along channel
        idx_y = (idx // p).view(b, 1, 1, h, w).float()
        idx_x = (idx % p).view(b, 1, 1, h, w).float()

        x = x.view(1,1,p,1,1).expand(1, 1, p, h, w).to('cuda').detach()
        y = y.view(1,p,1,1,1).expand(1, p, 1, h, w).to('cuda').detach()

        gauss_kernel = tr.exp(-((x-idx_x)**2 + (y-idx_y)**2) / (2 * sigma**2))
        gauss_kernel = gauss_kernel.view(b, p*p, h*w)#.permute(0,2,1).contiguous()

        return gauss_kernel * corr



    def match_to_flow_soft(self, match, k, h,w, window=2, temperature=1, mode='softmax'):        
        b, c , s = match.size()     
#         match = self.L2normalize(match)
        region = w / window
        idx = tr.arange(h*w, dtype=tr.float32).to('cuda')
        idx_x = idx % w
        idx_x = idx_x.repeat(b,k,1).to('cuda')
        idx_y = tr.floor(idx / w)   
        idx_y = idx_y.repeat(b,k,1).to('cuda')

        soft_idx_x = idx_x[:,:1]
        soft_idx_y = idx_y[:,:1]
        displacement = (self.patch-1)/2
        
        topk_value, topk_idx = tr.topk(match, k, dim=1)    # (B*T-1, k, H*W)
        topk_value = topk_value.view(-1,k,h,w)
        
        if (mode=='argmax'):
            topk_idx_x = topk_idx.float() % self.patch
            topk_idx_y = tr.floor(topk_idx.float() / self.patch)
        
            flow_x = self.patch_dilation * (topk_idx_x - displacement)#/w 
            flow_y =self.patch_dilation * (topk_idx_y - displacement)#/w 
            
            grid_x = tr.clamp(idx_x + flow_x,0,w-1)
            grid_y = tr.clamp(idx_y + flow_y,0,h-1)           
            
        elif (mode =='softmax'):
            match = self.apply_gaussian_kernel(match, h, w, self.patch, sigma=5)
            match = match*temperature
            match_pre = self.soft_argmax(match)
            smax = match_pre           
            smax = smax.view(b,self.patch,self.patch,h,w)
            x_kernel = tr.arange(-displacement*self.patch_dilation, displacement*self.patch_dilation+1, step=self.patch_dilation, dtype=tr.float).to('cuda')
            y_kernel = tr.arange(-displacement*self.patch_dilation, displacement*self.patch_dilation+1, step=self.patch_dilation, dtype=tr.float).to('cuda')
            x_mult = x_kernel.expand(b,self.patch).view(b,self.patch,1,1)
            y_mult = y_kernel.expand(b,self.patch).view(b,self.patch,1,1)
            
            smax_x = smax.sum(dim=1, keepdim=False) #(b,w=k,h,w)
            smax_y = smax.sum(dim=2, keepdim=False) #(b,h=k,h,w)
            flow_x = (smax_x*x_mult).sum(dim=1, keepdim=True).view(-1,1,h*w) # (b,1,h,w)
            flow_y = (smax_y*y_mult).sum(dim=1, keepdim=True).view(-1,1,h*w) # (b,1,h,w)
            grid_x = tr.clamp(soft_idx_x + flow_x,0,w-1)
            grid_y = tr.clamp(soft_idx_y + flow_y,0,h-1)
            
        grid_x = 2*(grid_x / (w-1)) - 1 #(b,1,h*w)
        grid_y = 2*(grid_y / (h-1)) - 1 #(b,1,h*w)

        flow_x = (flow_x / (self.patch_dilation * displacement))
        flow_y = (flow_y / (self.patch_dilation * displacement))
            
        return flow_x, flow_y, tr.cat([grid_x, grid_y], dim=1), topk_value
    

    def index_to_tensor(self, feature1, feature2, flow, topk_value):
        b, c, h, w = feature1.size()    
        k = topk_value.size()[1]

        # restore original integer flow
        displacement = (self.patch-1) / 2
        flow = (flow * self.patch_dilation * displacement).view(-1,2*k,h*w)
#         flow_x1 = flow[:,:k,:]
#         flow_y1 = flow[:,k:,:]
        
        # produce index
        idx = tr.arange(h*w, dtype=tr.float32).to('cuda').detach()
        idx_x = idx % w
        idx_x = idx_x.repeat(b,1,1).to('cuda').detach()
        idx_y = tr.floor(idx / w)   
        idx_y = idx_y.repeat(b,1,1).to('cuda').detach()

        # get target feature index
        disp_x = idx_x + flow[:,:k,:] # flow x
        disp_y = idx_y + flow[:,k:,:] # flow y
        disp_x = tr.clamp(disp_x, 0, w-1)
        disp_y = tr.clamp(disp_y, 0, h-1)
        disp_xy = w * disp_y + disp_x
        
        # add batch offsets
        b_idx = tr.arange(0, b*h*w, h*w).unsqueeze(1).unsqueeze(1).to('cuda').detach()
        mat_idx = b_idx + disp_xy.long()

        # get topk features
        topk_feature = feature2.permute(0, 2, 3, 1).reshape(-1, c)[mat_idx].permute(0, 1, 3, 2).view(-1, c, h, w).contiguous() # (B*T-1, k, c, h, w)

        # get source features
        src_feature = feature1.reshape(-1, 1, c, h, w) # (B*T-1, 1, c, h, w)
    
        return src_feature, topk_feature        
        
    def _make_layer(self, block, planes, blocks, num_segments, stride=1):       
        downsample = None        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
            
        layers = []
        layers.append(block(self.inplanes, planes, num_segments, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            remainder =int( i % 3)
            layers.append(block(self.inplanes, planes, num_segments, remainder=remainder))
            
        return nn.Sequential(*layers)            
            
    def _make_non_local_layer(self, non_local_block, block, planes, blocks, non_local, num_segments, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_segments))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks-non_local-1):
            layers.append(block(self.inplanes, planes, num_segments))            
        for _ in range(non_local):
            layers.append(non_local_block(self.inplanes, planes, num_segments))

        layers.append(block(self.inplanes, planes, num_segments))
        return nn.Sequential(*layers)
    
    def flow_computation(self, x, mode='rep', pos=2, temperature=100):

        size = x.size()               
        x = x.view((-1, self.num_segments) + size[1:])        # N T C H W
        x = x.permute(0,2,1,3,4).contiguous() # B C T H W   
            
            
        # match to flow            
        if (mode == 'match'):
            window = 4 #pow(2, (4-pos)) 
            k = 1            
            temperature = temperature                    
            b,c,t,h,w = x.size()            
            t = t-1         

            x_pre = x[:,:,:-1].permute(0,2,1,3,4).contiguous().view(-1,c,h,w)
            x_post = x[:,:,1:].permute(0,2,1,3,4).contiguous().view(-1,c,h,w)
            
            match = self.matching_layer(x_pre, x_post)    # (B*T-1*group, H*W, H*W)
#             match2 = self.matching_layer(x_pre2,x_post2)
#             match = self.L2normalize(match)
            
            u, v, grid_1, match_v = self.match_to_flow_soft(match, k, h, w, window, temperature, 'argmax')
#             match2 = self.matching_layer(x_post, x_pre)
#             u_2, v_2, grid_2, match_v2 = self.match_to_flow_soft(match2, k, h, w, window, temperature, 'argmax')    
#             u2, v2, grid_2 = self.match_to_flow_soft(match.permute(0,2,1).contiguous(), 1, hi, wi, window, temperature, 'softmax')     
            flow_1 = tr.cat([u,v], dim=1).view(-1, 2*k, h, w)  #  (b, 2, h, w)      
#             flow_2 = flow_1
            flow_2 = flow_1 #tr.cat([u_2,v_2],dim=1).view(-1,2, h, w)
            grid_1 = grid_1.view(-1,2, h, w).permute(0,2,3,1).contiguous()
            grid_2 = grid_1 #grid_2.view(-1,2, h, w).permute(0,2,3,1).contiguous()        
#             grid_2 = grid_1    
    
        return flow_1, flow_2, grid_1, grid_2, match_v       
        
    def forward(self, x, temperature):
        input =x    
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
       
        x = self.layer1(x)                             
        x = self.layer2(x)          
#         # Flow
        if (self.flow_estimation == 1):  
            flow_1, flow_2, grid_1, grid_2, match_v = self.flow_computation(x, mode='match', temperature=temperature, pos=2)
            x = self.flow_refinement(flow_1,x, match_v)

        x = self.layer3(x)                           
        x = self.layer4(x)
        x = self.avgpool(x)    
        x = x.view(x.size(0), -1)    
                       
        x = self.fc1(x)      
        return x


def resnet18(pretrained=False, shift='TSM',num_segments = 8, flow_estimation=0, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):    
        model = ResNet(BasicBlock, BasicBlock, [2, 2, 2, 2], NonLocal, num_segments=num_segments , flow_estimation=flow_estimation, **kwargs)  
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k in new_state_dict):
                new_state_dict.update({k:v})      
                print ("%s layer has pretrained weights" % k)
        model.load_state_dict(new_state_dict)
    return model


def resnet34(pretrained=False, shift='TSM',num_segments = 8, flow_estimation=0,**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):
        model = ResNet(BasicBlock, BasicBlock, [3, 4, 6, 3], NonLocal, num_segments=num_segments , flow_estimation=flow_estimation,  **kwargs)        
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k in new_state_dict):
                new_state_dict.update({k:v})      
                print ("%s layer has pretrained weights" % k)
        model.load_state_dict(new_state_dict)
    return model


def resnet50(pretrained=False, shift='TSM',num_segments = 8, flow_estimation=0, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):    
        model = ResNet(Bottleneck, Bottleneck, [3, 4, 6, 3], NonLocal, num_segments=num_segments , flow_estimation=flow_estimation, **kwargs)          
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k in new_state_dict):
                new_state_dict.update({k:v})      
                print ("%s layer has pretrained weights" % k)
        model.load_state_dict(new_state_dict)
    return model


def resnet101(pretrained=False, shift='TSM',num_segments = 8, flow_estimation=0, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):    
        model = ResNet(Bottleneck, Bottleneck, [3, 4, 23, 3], NonLocal, num_segments=num_segments , flow_estimation=flow_estimation, **kwargs)          
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k in new_state_dict):
                new_state_dict.update({k:v})      
                print ("%s layer has pretrained weights" % k)
        model.load_state_dict(new_state_dict)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
