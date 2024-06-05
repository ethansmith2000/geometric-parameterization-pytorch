
from torch import nn
import math
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _reverse_repeat_tuple, _pair
from torch.nn import init

def hyperspherical_to_cartesian(angles, radius):
    '''
        angles: [n_in-1, n_out], each column is a directional vector represented by (n_in-1) angular parameters
        radius: [1, n_out] or scalar (broadcast), radius for each direction
        
        return n_out cartesian coordinates: [n_in, n_out]
    '''
    pad_shape = (1, angles.shape[-1])
    augmented_angles = torch.cat([torch.zeros(pad_shape, dtype=angles.dtype, device=angles.device), angles], dim=-2)
    cos_angles = torch.cos(augmented_angles)
    rearranged_cos_angles = torch.roll(cos_angles, -1, -2)
    
    sin_angles = torch.sin(augmented_angles)
    sin_angles[0] = torch.ones(pad_shape, dtype=sin_angles.dtype, device=angles.device)
    accumulated_sin_angles = torch.cumprod(sin_angles, dim=-2)
    
    outputs = radius * accumulated_sin_angles * rearranged_cos_angles
    return outputs


def cartesian_to_hyperspherical(x):
    '''
        x: [n_in, n_out], each column is a vector in cartesian coordinates
        
        return n_out sets of angles: [n_in-1, n_out] and n_out radius
    '''
    numerator = x[:-1, :]
    correction = torch.sqrt(torch.sum(x[-2:, :]**2, dim=0, keepdim=True))
    numerator[-1, :] += correction.squeeze(0)
    denominator = torch.sqrt(torch.cumsum(x[1:, :].flip(0)**2, dim=0).flip(0))
    angles = math.pi / 2.0 - torch.atan2(numerator, denominator)
    angles[-1, :] = 2.0 * angles[-1, :]
    radius = torch.norm(x, dim=0, keepdim=True)

    return angles, radius


class GeoLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.angles = nn.Parameter(torch.empty(in_features-1, out_features))
        self.Lambda = nn.Parameter(torch.empty(out_features)) if bias else None
        self.radius = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    #hyperspherical glorot init
    def reset_parameters(self):
        threshold = math.sqrt(6.0 / (self.in_features + self.out_features))
        weight_init = torch.rand(self.in_features, self.out_features) * 2.0 * threshold - threshold
        angles, radius = cartesian_to_hyperspherical(weight_init)
        self.angles.data.copy_(angles)
        self.radius.data.copy_(radius.squeeze())
        if self.Lambda is not None:
            self.Lambda.data.zero_()

    def forward(self, input):
        weight = hyperspherical_to_cartesian(self.angles, radius=torch.tensor(1.0).to(input.device).to(input.dtype))
        out = F.linear(input, weight.T, self.Lambda)
        out = out * self.radius
        return out


class GeoLinear1D(nn.Module):

    def __init__(self, out_features, bias=True):
        super().__init__()
        self.in_features = 1
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(1, out_features))
        self.Lambda = nn.Parameter(torch.empty(out_features)) if bias else None
        self.radius = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    #hyperspherical glorot init
    def reset_parameters(self):
        threshold = math.sqrt(6.0 / (self.in_features + self.out_features))
        weight_init = torch.rand(self.in_features, self.out_features) * 2.0 * threshold - threshold
        bias_init = 0.0
        radius = torch.abs(weight_init).squeeze()
        lambda_init = bias_init / radius
        self.weight.data.copy_(weight_init)
        self.radius.data.copy_(radius)
        if self.Lambda is not None:
            self.Lambda.data.copy_(lambda_init)


    def forward(self, input):
        with torch.no_grad():
            weight = self.weight / torch.abs(self.weight)
        out = F.linear(input, weight.T, self.Lambda)
        out = out * self.radius
        return out


class GeoConvNd(nn.Module):

    def __init__(self, 
                in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 transposed,
                 output_padding,
                 groups,
                 bias,
                 padding_mode,
                 device=None,
                 dtype=None
                ):

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}")
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError(f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                   range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)


        if transposed:
            in_dim = in_channels
            out_dim = out_channels // groups
        else:
            in_dim = in_channels // groups
            out_dim = out_channels
        for k in kernel_size:
            in_dim *= k
        # subtract one dim because we get it from the hypersphere transform
        self.weight = nn.Parameter(torch.empty((in_dim - 1, out_dim), **factory_kwargs))

        self.radius = nn.Parameter(torch.empty(1, out_dim, 1, 1, **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'



class GeoConv2d(GeoConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

        self.reset_parameters()


    def reset_parameters(self):
        left, right = self.weight.shape[0]+1, self.weight.shape[1]
        threshold = math.sqrt(6.0 / (left + right))
        weight_init = torch.rand(left, right) * 2.0 * threshold - threshold
        angles, radius = cartesian_to_hyperspherical(weight_init)
        self.weight.data.copy_(angles)
        self.radius.data.copy_(radius.squeeze()[None,:,None,None])
        if self.bias is not None:
            self.bias.data.zero_()

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        weight = hyperspherical_to_cartesian(self.weight, radius=torch.tensor(1.0).to(input.device).to(input.dtype))
        kernel = weight.reshape(self.in_channels, *self.kernel_size, self.out_channels)
        kernel = kernel.permute(3, 0, 1, 2)
        return self._conv_forward(input, kernel, self.bias) * self.radius