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

class Matching_layer(nn.Module):
    def __init__(self, ks, patch, stride, pad, patch_dilation):
        super(Matching_layer, self).__init__()
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
    
class Flow_refinement(nn.Module):
    def init(self, num_segments, expansion = 1, pos=2):
        super(Flow_refinement, self).__init__()
        self.num_segments = num_segments
        self.expansion = expansion
        self.pos = pos
        self.out_channel = 64*(2**(self.pos-1))*self.expansion

        self.c1 = 16
        self.c2 = 32
        self.c3 = 64

        self.conv1 = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=7, stride=1, padding=3, groups=3, bias=False),
        nn.BatchNorm2d(3),
        nn.ReLU(),
        nn.Conv2d(3, self.c1, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(self.c1),
        nn.ReLU()
        )
        self.conv2 = nn.Sequential(
        nn.Conv2d(self.c1, self.c1, kernel_size=3, stride=1, padding=1, groups=self.c1, bias=False),
        nn.BatchNorm2d(self.c1),
        nn.ReLU(),
        nn.Conv2d(self.c1, self.c2, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(self.c2),
        nn.ReLU()
        )
        self.conv3 = nn.Sequential(
        nn.Conv2d(self.c2, self.c2, kernel_size=3, stride=1, padding=1, groups=self.c2, bias=False),
        nn.BatchNorm2d(self.c2),
        nn.ReLU(),
        nn.Conv2d(self.c2, self.c3, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(self.c3),
        nn.ReLU()
        )
        self.conv4 = nn.Sequential(
        nn.Conv2d(self.c3, self.c3, kernel_size=3, stride=1, padding=1, groups=self.c3, bias=False),
        nn.BatchNorm2d(self.c3),
        nn.ReLU(),
        nn.Conv2d(self.c3, self.out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(self.out_channel),
        nn.ReLU()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res, match_v):
        if match_v is not None:
            x = tr.cat([x, match_v], dim=1)
        _, c, h, w = x.size()
        x = x.view(-1,self.num_segments-1,c,h,w)

        x = tr.cat([x,x[:,-1:,:,:,:]], dim=1) ## (b,t,3,h,w)
        x = x.view(-1,c,h,w)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + res

        return x
    
class ResNet(nn.Module):

    def __init__(self, block, block2, layers, num_segments, flow_estimation, num_classes=1000, zero_init_residual=False):
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
                                                                   
      
        ## MotionSqueeze
        self.patch= 15
        self.patch_dilation =1
        self.matching_layer = Matching_layer(ks=1, patch=self.patch, stride=1, pad=0, patch_dilation=self.patch_dilation)                              
        self.flow_refinement = Flow_refinement(num_segments=num_segments, expansion=block.expansion,pos=2)      
        self.soft_argmax = nn.Softmax(dim=1)
             
       
        self.layer1 = self._make_layer(block, 64, layers[0], num_segments=num_segments)
        self.layer2 = self._make_layer(block, 128, layers[1],  num_segments=num_segments, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2],  num_segments=num_segments, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3],  num_segments=num_segments, stride=2)       
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc1 = nn.Linear(512 * block.expansion, num_classes)                   
        self.fc1 = nn.Conv1d(512*block.expansion, num_classes, kernel_size=1, stride=1, padding=0,bias=True)         
        
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

    def match_to_flow_soft(self, match, k, h,w, temperature=1, mode='softmax'):        
        b, c , s = match.size()     
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
        
#         grid_x = tr.clamp(soft_idx_x + flow_x,0,w-1)
#         grid_y = tr.clamp(soft_idx_y + flow_y,0,h-1)            
#         grid_x = 2*(grid_x / (w-1)) - 1 #(b,1,h*w)
#         grid_y = 2*(grid_y / (h-1)) - 1 #(b,1,h*w)

        flow_x = (flow_x / (self.patch_dilation * displacement))
        flow_y = (flow_y / (self.patch_dilation * displacement))
            
        return flow_x, flow_y, topk_value          
        
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
    
    def flow_computation(self, x, pos=2, temperature=100):

        size = x.size()               
        x = x.view((-1, self.num_segments) + size[1:])        # N T C H W
        x = x.permute(0,2,1,3,4).contiguous() # B C T H W   
                        
        # match to flow            
        k = 1            
        temperature = temperature                    
        b,c,t,h,w = x.size()            
        t = t-1         

        x_pre = x[:,:,:-1].permute(0,2,1,3,4).contiguous().view(-1,c,h,w)
        x_post = x[:,:,1:].permute(0,2,1,3,4).contiguous().view(-1,c,h,w)
            
        match = self.matching_layer(x_pre, x_post)    # (B*T-1*group, H*W, H*W)          
        u, v, confidence = self.match_to_flow_soft(match, k, h, w, temperature)
        flow = tr.cat([u,v], dim=1).view(-1, 2*k, h, w)  #  (b, 2, h, w)            
                
        # backward flow
#             match2 = self.matching_layer(x_post, x_pre)
#             u_2, v_2, confidence_2 = self.match_to_flow_soft(match2, k, h, w,temperature)       
#             flow_2 = tr.cat([u_2,v_2],dim=1).view(-1,2, h, w)   
    
        return flow, confidence     
        
    def forward(self, x, temperature):
        input =x    
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
       
        x = self.layer1(x)                             
        x = self.layer2(x)          
        
        # Flow
        if (self.flow_estimation == 1):  
            flow_1, match_v = self.flow_computation(x, temperature=temperature, pos=2)
            x = self.flow_refinement(flow_1,x, match_v)

        x = self.layer3(x)                           
        x = self.layer4(x)
        x = self.avgpool(x)    
        x = x.view(x.size(0), -1,1)    
                       
        x = self.fc1(x)      
        return x


def resnet18(pretrained=False, shift='TSM',num_segments = 8, flow_estimation=0, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):    
        model = ResNet(BasicBlock, BasicBlock, [2, 2, 2, 2], num_segments=num_segments , flow_estimation=flow_estimation, **kwargs)  
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k in new_state_dict):
                new_state_dict.update({k:v})      
#                 print ("%s layer has pretrained weights" % k)
        model.load_state_dict(new_state_dict)
    return model


def resnet34(pretrained=False, shift='TSM',num_segments = 8, flow_estimation=0,**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):
        model = ResNet(BasicBlock, BasicBlock, [3, 4, 6, 3],num_segments=num_segments , flow_estimation=flow_estimation,  **kwargs)        
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k in new_state_dict):
                new_state_dict.update({k:v})      
#                 print ("%s layer has pretrained weights" % k)
        model.load_state_dict(new_state_dict)
    return model


def resnet50(pretrained=False, shift='TSM',num_segments = 8, flow_estimation=0, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):    
        model = ResNet(Bottleneck, Bottleneck, [3, 4, 6, 3],num_segments=num_segments , flow_estimation=flow_estimation, **kwargs)          
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k in new_state_dict):
                new_state_dict.update({k:v})      
#                 print ("%s layer has pretrained weights" % k)
        model.load_state_dict(new_state_dict)
    return model


def resnet101(pretrained=False, shift='TSM',num_segments = 8, flow_estimation=0, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):    
        model = ResNet(Bottleneck, Bottleneck, [3, 4, 23, 3],num_segments=num_segments , flow_estimation=flow_estimation, **kwargs)          
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k in new_state_dict):
                new_state_dict.update({k:v})      
#                 print ("%s layer has pretrained weights" % k)
        model.load_state_dict(new_state_dict)
    return model
