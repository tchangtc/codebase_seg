import torch 
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'LayerNorm',
    
]

class LayerNorm(nn.Module):
    r""" LayerNorm that support two data formats: channels_last (default) or channels_first.
    The ordering of dimensions in the inputs. channels_last corresponds to inputs with shape
    (batch_size, height, width, channels), while channels_first corresponds to input with
    (channels, batch_size, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super(LayerNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplemented
        self.normalized_shape = (normalized_shape,)
    


    """
    reference: https://blog.csdn.net/weixin_45019478/article/details/115027728
    The difference between F.layer_norm and nn.LayerNorm:
    F.layer_norm: exists no learnable parameters
    nn.LayerNorm: exists learnable paarameters
    when "elementwise_affine" is set to be False, nn.LayerNorm becomes F.layer_norm。
    """
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]
            return x