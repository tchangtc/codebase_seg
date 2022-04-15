import torch
import torch.nn as nn
import torch.nn.functional as F

def passthrough(x, **kwargs):

    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary for good performance

class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        # out = self.bn1(self.conv1(self.relu1(x)))
        out = self.relu1(self.bn1(self.conv1(x)))
        return out

def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    
    return nn.Sequential(*layers)

class InputTransition(nn.Module):
    def __init__(self, in_ch, out_ch, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(out_ch)
        self.relu1 = ELUCons(elu, out_ch)
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split in put into 16 channels
        num = int(self.out_ch / self.in_ch)
        # print(self.out_ch)
        # print(self.in_ch)
        # print(num)
        x_16 = x.repeat(1, num, 1, 1, 1)
        out = self.relu1(torch.add(out, x_16))

        return out
        # # do we want a PRELU here as well?
        # x1 = x
        # x = self.conv1(x)
        # out = self.bn1(x)
        # # print(out.shape) # 1, 16, 16, 256, 256
        # # print(x.shape)
        # # split input in to 16 channels
        # x16 = torch.cat((x1, x1, x1, x1, x1, x1, x1, x1,
        #                  x1, x1, x1, x1, x1, x1, x1, x1), 1)
        # # print("x16", x16.shape)
        # # print("out:", out.shape)
        # # assert 1>3
        # out = self.relu1(torch.add(out, x16))
        # # print(out.shape) # 1, 16, 16, 256, 256
        # # assert 1>3
        # return out

class DownTransition(nn.Module):
    def __init__(self, in_ch, nConvs, elu, scale=2, dropout=False):
        super(DownTransition, self).__init__()
        out_ch = in_ch * 2
        self.down_conv = nn.Conv3d(in_ch, out_ch, kernel_size=scale, stride=scale)
        self.bn1 = ContBatchNorm3d(out_ch)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, out_ch)
        self.relu2 = ELUCons(elu, out_ch)

        if dropout:
            self.do1 = nn.Dropout3d()
        
        self.ops = _make_nConv(out_ch, nConvs, elu)
                 # _make_nConv(nchan, depth, elu):

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))

        return out

class UpTransition(nn.Module):
    def __init__(self, in_ch, out_ch, nConvs, elu, scale=2, dropout=False):
        super(UpTransition, self).__init__()
        
        self.up_conv = nn.ConvTranspose3d(in_ch, out_ch // 2, kernel_size=scale, stride=scale)
        self.bn1 = ContBatchNorm3d(out_ch // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, out_ch // 2)
        self.relu2 = ELUCons(elu, out_ch)
        
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(out_ch, nConvs, elu)
    """
        The forward func has some differences in Vnet paper
    """
    def forward(self, x, skipx):
        # print('x.shape: {} and skipx.shape: {}'.format(x.shape, skipx.shape))
        out = self.do1(x)   # passthrough 
        skipxdo = self.do2(skipx)   # encoder part
        out = self.relu1(self.bn1(self.up_conv(out)))   # through up_conv  
        xcat = torch.cat((out, skipxdo), dim=1)     # skip-connection
        out = self.ops(xcat)    # 5 * 5 convolution 
        # print('out.shape: {} and xcat.shape: {}'.format(out.shape, xcat.shape))
        out = self.relu2(torch.add(out, xcat))  # Here, 'xcat' means the out that passed through 'up_conv' cat the skip one, 
                                                # and, 'out' means xcat through 5*5 conv.
        return out

class OutputTransition(nn.Module):
    def __init__(self, in_ch, out_ch, elu, nll):
        super(OutputTransition, self).__init__()

        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=1)
        self.relu1 = ELUCons(elu, out_ch)
    
    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out 

class VNet(nn.Module):
    def __init__(self, in_ch, out_ch, scale, base_ch=16, elu=True, nll=False):
        super(VNet, self).__init__()

        self.in_tr = InputTransition(in_ch, base_ch, elu)
        self.down_tr32 = DownTransition(base_ch, 1, elu, scale=scale[0])
        self.down_tr64 = DownTransition(base_ch*2, 2, elu, scale=scale[1])
        self.down_tr128 = DownTransition(base_ch*4, 3, elu, dropout=True, scale=scale[2])
        self.down_tr256 = DownTransition(base_ch*8, 2, elu, dropout=True, scale=scale[3])

        self.up_tr256 = UpTransition(base_ch*16, base_ch*16, 2, elu, dropout=True,scale=scale[3])
        self.up_tr128 = UpTransition(base_ch*16, base_ch*8, 2, elu, dropout=True, scale=scale[2])
        self.up_tr64 = UpTransition(base_ch*8, base_ch*4, 1, elu, scale=scale[1])
        self.up_tr32 = UpTransition(base_ch*4, base_ch*2, 1, elu, scale=scale[0])

        self.out_tr = OutputTransition(base_ch*2, out_ch, elu, nll)
    
    def forward(self, x):

        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)

        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        
        out = self.out_tr(out)

        return out


# def test():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(device)
#     img = torch.rand((1,2,256,256,64)).to(device)
#     print(img.shape)
#     model = VNet(in_ch=2, out_ch=18, scale=[[1,2,2], [2,2,2], [2,2,2], [2,2,2]], base_ch=16).to(device)
#     pred = model(img)
#     print(pred.shape)

# test()