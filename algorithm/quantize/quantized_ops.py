"""
In this file, we implement quantized ops to simulate the device-side inference of quantized models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

sys.path.append("..")

USE_FP_SCALE = True


#############################
########### utils ###########
#############################

def to_np(x):  # cast x to np array if it is a number
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return np.array([x])


def to_pt(x):
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.float64:
            return x.float()
        else:
            return x.float()  # NOTE: for backward, we need everything float
    elif isinstance(x, np.ndarray):
        org_dtype = x.dtype
        x = torch.from_numpy(x)
        if org_dtype in [np.int64, np.int32, np.int8]:
            return x.float()  # NOTE: for backward, we need everything float
        elif org_dtype in [np.float32, np.float64]:
            return x.float()
        else:
            raise NotImplementedError(org_dtype)
    else:
        return to_pt(np.array(x))


#############################
####### quantized ops #######
#############################

class QuantizedAvgPool(nn.Module):
    def __init__(self):
        super(QuantizedAvgPool, self).__init__()

    def forward(self, x):
        x = x.float()
        x = x.mean([-1, -2], keepdim=True).round().int()
        return x


class QuantizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 zero_x=0, zero_w=0, zero_y=0,
                 effective_scale=None,
                 significand=1, channel_shift=0,
                 w_bit=8, a_bit=None,
                 ):
        super(QuantizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                              padding, dilation, groups, bias, padding_mode)
        self.register_buffer('zero_x', to_pt(zero_x))
        self.register_buffer('zero_w', to_pt(zero_w))
        self.register_buffer('zero_y', to_pt(zero_y))
        self.register_buffer('effective_scale', effective_scale)
        self.significand = significand
        self.channel_shift = channel_shift

        self.w_bit = w_bit
        self.a_bit = a_bit if a_bit is not None else w_bit

    def forward(self, x):
        # assume x and weight are both in int8
        weight = self.weight.int()  # - self.zero_w.view(-1, 1, 1, 1)
        # CXD: zero_x gets passed in from the configuration file.
        x = x.int() - self.zero_x # CXD: shift x to center around the zero value

        """ 
        CXD: F.conv2d will create either a depthwise or a non-depthwise conv2d layer.
        If it's depthwise convolutional, then the groups will match the number of input channels, nothing more.
        In short, don't need to worry about making a new resnet model.
        https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/5
        """
        out = F.conv2d(x.float(), weight.float(), None, self.stride, self.padding, self.dilation,
                       self.groups).round().int()
        out = out + self.bias.int().view(1, -1, 1, 1)
        if self.effective_scale is not None:
            """ CXD: if effective_scale is set (picked up from x*w / y) then 
            return out as a double (?) multiplied by x*w/y, rounded back to an int."""
            out = (out.double() * self.effective_scale.view(1, -1, 1, 1)).round().int()
        else:
            out = out.type(torch.int64)
            out = out * self.significand.view(1, -1, 1, 1)
            # add nudge
            # CXD: if elt in out is positive, add 1 shifted left 30 bits ( * 2^30) [kind of big no?]
            out[out >= 0] += (1 << 30)
            # CXD: if elt is negative, add negative version
            out[out < 0] += (1 - (1 << 30))
            # CXD: basically, the above just increases the magnitude / range

            out = out // (1 << 31) # CXD: and then divide out the extra added value. like x/a + 1
            out = out.cpu().numpy()

            shift = (-self.channel_shift).view(1, -1, 1, 1) # CXD: generally channel_shift is 0
            shift = shift.cpu().numpy()

            mask = ((1 << shift) - 1)
            # CXD: mask is just 0 matrix usually? Seems to be unassigned. * 2^shift
            remainder = mask & out # CXD: remainder would then be mostly 0?
            threshold = mask >> 1 # CXD: Still 0...? divided by two (half of mask)
            remainder[out < 0] -= 1 # CXD: mat of 0, -1?
            out = out >> shift # CXD: multiply out by 2^shift, but shift is 0 so nothing
            out[remainder > threshold] += 1 # CXD: then add one...nowhere usually...?

            out = torch.from_numpy(out)
            # CXD: If channel_shift is nonzero, would multiply out by 2**channel_shift, then add 1 to any out values > threshold (1/2)
        out = out + self.zero_y # CXD: add zero_y back in
        return out.clamp(- 2 ** (self.a_bit - 1), 2 ** (self.a_bit - 1) - 1) # CXD: return out, limited to [-2^(n-1), 2^(n-2)]


class QuantizedElementwise(nn.Module):
    def __init__(self, operator, zero_x1, zero_x2, zero_y, scale_x1, scale_x2, scale_y):
        super().__init__()
        self.operator = operator
        assert operator in ['add', 'mult']
        self.register_buffer('zero_x1', to_pt(zero_x1))
        self.register_buffer('zero_x2', to_pt(zero_x2))
        self.register_buffer('zero_y', to_pt(zero_y))
        self.register_buffer('scale_x1', to_pt(scale_x1))
        self.register_buffer('scale_x2', to_pt(scale_x2))
        self.register_buffer('scale_y', to_pt(scale_y))

    def forward(self, x1, x2):
        x1 = (x1.int() - self.zero_x1) * self.scale_x1
        x2 = (x2.int() - self.zero_x2) * self.scale_x2

        if self.operator == 'add':
            out = x1 + x2
        elif self.operator == 'mult':
            out = x1 * x2
        else:
            raise NotImplementedError
        out = (out.double() / self.scale_y).round().int()
        out = out + self.zero_y
        return out


class QuantizedSE(nn.Module):
    def __init__(self, fc, q_mult, a_bit=8):
        super().__init__()
        self.fc = fc
        self.avg_pool = QuantizedAvgPool()
        self.q_mult = q_mult

        self.a_bit = a_bit

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc(out)
        x = self.q_mult(x, out)
        return x.clamp(- 2 ** (self.a_bit - 1), 2 ** (self.a_bit - 1) - 1)


class QuantizedMbBlock(nn.Module):
    def __init__(self, conv, q_add=None, residual_conv=None, a_bit=8):
        super().__init__()
        self.conv = conv
        self.q_add = q_add
        self.residual_conv = residual_conv

        self.a_bit = a_bit

    def forward(self, x):
        out = self.conv(x)
        if self.q_add is not None:
            if self.residual_conv is not None:
                x = self.residual_conv(x)
            out = self.q_add(x, out)
            return out.clamp(- 2 ** (self.a_bit - 1), 2 ** (self.a_bit - 1) - 1)
        else:
            return out

