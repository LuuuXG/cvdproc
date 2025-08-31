# Code from: /disk/hz459/coding/ms_lesion_segmentatation/lesion_seg/

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn import Parameter


class convBlockVGG_ND(nn.Module):

    def __init__(self, 
        num_channels = [9, 32], 
        is_batchnorm = True,
        dimension = 3,
        kernel_size = 3,
        stride = 1,
        padding = 1
    ):

        super(convBlockVGG_ND, self).__init__()

        self.num_channels = num_channels
        self.is_batchnorm = is_batchnorm
        self.dimension = dimension
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if self.dimension == 1:
            conv_ND = nn.Conv1d
            batchNorm_ND = nn.BatchNorm1d
        elif self.dimension == 2:
            conv_ND = nn.Conv2d
            batchNorm_ND = nn.BatchNorm2d
        elif self.dimension == 3:
            conv_ND = nn.Conv3d
            batchNorm_ND = nn.BatchNorm3d

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                conv_ND(self.num_channels[0], self.num_channels[1], self.kernel_size, self.stride, self.padding), 
                batchNorm_ND(self.num_channels[1]), 
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                conv_ND(self.num_channels[1], self.num_channels[1], self.kernel_size, self.stride, self.padding), 
                batchNorm_ND(self.num_channels[1]), 
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                conv_ND(self.num_channels[0], self.num_channels[1], self.kernel_size, self.stride, self.padding), 
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                conv_ND(self.num_channels[1], self.num_channels[1], self.kernel_size, self.stride, self.padding), 
                nn.ReLU(inplace=True)
            )

    def forward(self, inputs):

        outputs = self.conv2(self.conv1(inputs))

        return outputs

class unetConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, is_batchnorm, conv_type = 'vgg'):
        
        super(unetConv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_batchnorm = is_batchnorm
        self.conv_type = conv_type

        self.convBlocks = []

        if self.conv_type == 'vgg':
            self.convBlock1 = convBlockVGG_ND(
                num_channels = [self.in_channels, self.out_channels],
                is_batchnorm = self.is_batchnorm,
                dimension = 3
            )
            self.convBlocks.append(self.convBlock1)

        self.convolution = nn.Sequential(*self.convBlocks)

    def forward(self, inputs):

        outputs = self.convolution(inputs)

        return outputs

class upsample3d(nn.Module):

    def __init__(self, in_size, out_size, is_deconv = True, is_hpool = True):
        
        super(upsample3d, self).__init__()
        
        if is_deconv:
            if is_hpool:            
                self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size = 2, stride = 2, padding = 0)
            else:
                self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size = (1, 2, 2), stride = (1, 2, 2), padding = 0)
        else:
            self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear')

    def forward(self, x):

        return self.up(x)

class pad3d(nn.Module):

    def __init__(self):

        super(pad3d, self).__init__()
    
    def forward(self, leftIn, rightIn):
        
        rShape = rightIn.size()
        lShape = leftIn.size()
        padding = (lShape[4]-rShape[4], 0, lShape[3]-rShape[3], 0, lShape[2]-rShape[2], 0)
        
        pad = torch.nn.ConstantPad3d(padding, 0)
        rightIn = pad(rightIn)

        return leftIn, rightIn

class cat3d(nn.Module):

    def __init__(self):

        super(cat3d, self).__init__()
    
    def forward(self, leftIn, rightIn):

        lrCat = torch.cat([leftIn, rightIn], 1)

        return lrCat

class padConcate3d(nn.Module):

    def __init__(self):

        super(padConcate3d, self).__init__()

    def forward(self, leftIn, rightIn):
        
        rShape = rightIn.size()
        lShape = leftIn.size()
        padding = (lShape[4]-rShape[4], 0, lShape[3]-rShape[3], 0, lShape[2]-rShape[2], 0)
        
        pad = torch.nn.ConstantPad3d(padding, 0)
        rightIn = pad(rightIn)

        lrCat = torch.cat([leftIn, rightIn], 1)

        return lrCat 
        

class unetUp3d(nn.Module):

    def __init__(self, in_size, out_size, is_deconv, is_hpool = True):
        
        super(unetUp3d, self).__init__()
        
        self.conv = unetConv3d(in_size, out_size, False)
        if is_deconv:
            if is_hpool:            
                self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size = 2, stride = 2, padding = 0)
            else:
                self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size = (1, 2, 2), stride = (1, 2, 2), padding = 0)
        else:
            self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear')

    def forward(self, leftIn, rightIn):
        
        rightIn = self.up(rightIn)
        rShape = rightIn.size()
        lShape = leftIn.size()
        padding = (lShape[4]-rShape[4], 0, lShape[3]-rShape[3], 0, lShape[2]-rShape[2], 0)
        
        pad = torch.nn.ConstantPad3d(padding, 0)
        rightIn = pad(rightIn)

        lrCat = torch.cat([leftIn, rightIn], 1)
        output = self.conv(lrCat)
        return output 


class sliceAttEncDecBlockDepth(nn.Module):

    def __init__(self, att_type = 0, with_ori = False):
        
        super(sliceAttEncDecBlockDepth, self).__init__()
        self.att_type = att_type
        self.with_ori = with_ori
        self.beta_left = Parameter(torch.tensor(1.0, requires_grad = True))
        self.beta_right = Parameter(torch.tensor(1.0, requires_grad = True))

    def forward(self, left, right):

        batch_size = left.size()[0]
        depth_size = left.size()[2]

        # n * c * d * h * w
        left_ori = left.permute(0, 2, 1, 3, 4) # n * d * c * h * w
        right_ori = right.permute(0, 2, 1, 3, 4) # n * d * c * h * w 
        left_ = left_ori.contiguous().view(batch_size, depth_size, -1) # n * d * chw
        right_ = right_ori.contiguous().view(batch_size, depth_size, -1) # n * d * chw
        
        out_left = out_right= None

        if self.att_type == 0:
            
            left_t  = left_.permute(0, 2, 1) # n * chw * d
            attention_left = torch.matmul(right_, left_t) # n * d * d
            attention_left = F.softmax(attention_left, dim = -1) # n * d * d
            
            right_t  = right_.permute(0, 2, 1) # n * chw * d
            attention_right = torch.matmul(left_, right_t) # n * d * d
            attention_right = F.softmax(attention_right, dim = -1) # n * d * d
            
            out_left = torch.matmul(attention_left, left_) # n * d * chw
            out_right = torch.matmul(attention_right, right_) # n * d * chw
        else: 

            attention = None

            if self.att_type == 1:
                left_t  = left_.permute(0, 2, 1) # n * chw * d
                attention = torch.matmul(right_, left_t) # n * d * d
            elif self.att_type == 2:
                right_t  = right_.permute(0, 2, 1) # n * chw * d
                attention = torch.matmul(left_, right_t) # n * d * d

            attention = F.softmax(attention, dim = -1) # n * d * d, density function along the row 

            out_left = torch.matmul(attention, left_) # n * d * chw
            out_right = torch.matmul(attention, right_) # n * d * chw

        if self.with_ori:
            out_left = self.beta_left * out_left + left_ # n * d * chw
            out_right = self.beta_right * out_right + right_ # n * d * chw
        else:
            out_left = self.beta_left * out_left # n * d * chw
            out_right = self.beta_right * out_right # n * d * chw

        out_left = out_left.contiguous().view(batch_size, depth_size, *left_ori.size()[2:])      
        # n * d * c * h * w
        out_right = out_right.contiguous().view(batch_size, depth_size, *right_ori.size()[2:])      
        # n * d * c * h * w
        out_left = out_left.permute(0, 2, 1, 3, 4)
        # n * c * d * h * w
        out_right = out_right.permute(0, 2, 1, 3, 4)
        # n * c * d * h * w

        return out_left, out_right

class sliceAttEncDecBlockHeight(nn.Module):

    def __init__(self, att_type = 0, with_ori = False):
        
        super(sliceAttEncDecBlockHeight, self).__init__()
        self.att_type = att_type
        self.with_ori = with_ori
        self.beta_left = Parameter(torch.tensor(1.0, requires_grad = True))
        self.beta_right = Parameter(torch.tensor(1.0, requires_grad = True))

    def forward(self, left, right):

        batch_size = left.size()[0]
        height_size = left.size()[3]

        # n * c * d * h * w
        left_ori = left.permute(0, 3, 2, 1, 4) # n * h * d * c * w
        right_ori = right.permute(0, 3, 2, 1, 4) # n * h * d * c * w 
        left_ = left_ori.contiguous().view(batch_size, height_size, -1) # n * h * dcw
        right_ = right_ori.contiguous().view(batch_size, height_size, -1) # n * h * dcw
        
        out_left = out_right= None

        if self.att_type == 0:
            
            left_t  = left_.permute(0, 2, 1) # n * dcw * h
            attention_left = torch.matmul(right_, left_t) # n * h * h
            attention_left = F.softmax(attention_left, dim = -1) # n * h * h
            
            right_t  = right_.permute(0, 2, 1) # n * dcw * h
            attention_right = torch.matmul(left_, right_t) # n * h * h
            attention_right = F.softmax(attention_right, dim = -1) # n * h * h
            
            out_left = torch.matmul(attention_left, left_) # n * h * dcw
            out_right = torch.matmul(attention_right, right_) # n * h * dcw
        else: 

            attention = None

            if self.att_type == 1:
                left_t  = left_.permute(0, 2, 1) # n * dcw * h
                attention = torch.matmul(right_, left_t) # n * h * h
            elif self.att_type == 2:
                right_t  = right_.permute(0, 2, 1) # n * dcw * h
                attention = torch.matmul(left_, right_t) # n * h * h

            attention = F.softmax(attention, dim = -1) # n * h * h, density function along the row 

            out_left = torch.matmul(attention, left_) # n * h * dcw
            out_right = torch.matmul(attention, right_) # n * h * dcw

        if self.with_ori:
            out_left = self.beta_left * out_left + left_ # n * h * dcw
            out_right = self.beta_right * out_right + right_ # n * h * dcw
        else:
            out_left = self.beta_left * out_left # n * h * dcw
            out_right = self.beta_right * out_right # n * h * dcw

        out_left = out_left.contiguous().view(batch_size, height_size, *left_ori.size()[2:])      
        # n * h * d * c * w
        out_right = out_right.contiguous().view(batch_size, height_size, *right_ori.size()[2:])      
        # n * h * d * c * w
        out_left = out_left.permute(0, 3, 2, 1, 4)
        # n * c * d * h * w
        out_right = out_right.permute(0, 3, 2, 1, 4)
        # n * c * d * h * w

        return out_left, out_right

class sliceAttEncDecBlockWidth(nn.Module):

    def __init__(self, att_type = 0, with_ori = False):
        
        super(sliceAttEncDecBlockWidth, self).__init__()
        self.att_type = att_type
        self.with_ori = with_ori
        self.beta_left = Parameter(torch.tensor(1.0, requires_grad = True))
        self.beta_right = Parameter(torch.tensor(1.0, requires_grad = True))

    def forward(self, left, right):

        batch_size = left.size()[0]
        width_size = left.size()[4]

        # n * c * d * h * w
        left_ori = left.permute(0, 4, 2, 3, 1) # n * w * d * h * c
        right_ori = right.permute(0, 4, 2, 3, 1) # n * w * d * h * c 
        left_ = left_ori.contiguous().view(batch_size, width_size, -1) # n * w * dhc
        right_ = right_ori.contiguous().view(batch_size, width_size, -1) # n * w * dhc
        
        out_left = out_right= None

        if self.att_type == 0:
            
            left_t  = left_.permute(0, 2, 1) # n * dhc * w
            attention_left = torch.matmul(right_, left_t) # n * w * w
            attention_left = F.softmax(attention_left, dim = -1) # n * w * w
            
            right_t  = right_.permute(0, 2, 1) # n * dhc * w
            attention_right = torch.matmul(left_, right_t) # n * w * w
            attention_right = F.softmax(attention_right, dim = -1) # n * w * w
            
            out_left = torch.matmul(attention_left, left_) # n * w * dhc
            out_right = torch.matmul(attention_right, right_) # n * w * dhc
        else: 

            attention = None

            if self.att_type == 1:
                left_t  = left_.permute(0, 2, 1) # n * dhc * w
                attention = torch.matmul(right_, left_t) # n * w * w
            elif self.att_type == 2:
                right_t  = right_.permute(0, 2, 1) # n * dhc * w
                attention = torch.matmul(left_, right_t) # n * w * w

            attention = F.softmax(attention, dim = -1) # n * w * w, density function along the row 

            out_left = torch.matmul(attention, left_) # n * w * dhc
            out_right = torch.matmul(attention, right_) # n * w * dhc

        if self.with_ori:
            out_left = self.beta_left * out_left + left_ # n * h * dcw
            out_right = self.beta_right * out_right + right_ # n * h * dcw
        else:
            out_left = self.beta_left * out_left # n * h * dcw
            out_right = self.beta_right * out_right # n * h * dcw

        out_left = out_left.contiguous().view(batch_size, width_size, *left_ori.size()[2:])      
        # n * w * d * h * c
        out_right = out_right.contiguous().view(batch_size, width_size, *right_ori.size()[2:])      
        # n * w * d * h * c
        out_left = out_left.permute(0, 4, 2, 3, 1)
        # n * c * d * h * w
        out_right = out_right.permute(0, 4, 2, 3, 1)
        # n * c * d * h * w

        return out_left, out_right

class sliceAttEncDecBlockShakeDir(nn.Module):

    '''
    slice_dir: 
    three different directions
    1. depth denotes the depth direction
    2. height denotes the height direction
    3. width denotes the width direction
    '''

    def __init__(self, slice_dir = 0, att_type = 0, shake_config = [False, False], with_ori = False):
        
        super(sliceAttEncDecBlockShakeDir, self).__init__()

        self.slice_dir = slice_dir
        self.att_type = att_type
        self.shake_config = shake_config
        self.with_ori = with_ori

        if self.slice_dir == 0:
            self.att = sliceAttEncDecBlockDepth(att_type = self.att_type, with_ori = self.with_ori)
        elif self.slice_dir == 1:
            self.att = sliceAttEncDecBlockHeight(att_type = self.att_type, with_ori = self.with_ori)            
        elif self.slice_dir == 2:
            self.att = sliceAttEncDecBlockWidth(att_type = self.att_type, with_ori = self.with_ori)                        
        else:
            self.att = sliceAttEncDecBlockDepth(att_type = self.att_type, with_ori = self.with_ori)

    def forward(self, left, right):

        leftAtt, rightAtt = self.att(left, right)

        if self.training:
            shake_config = self.shake_config
        else:
            shake_config = [False, False]

        alphaBetaLeft = get_alpha_beta_dir(shake_config, left.is_cuda)
        alphaBetaRight = get_alpha_beta_dir(shake_config, right.is_cuda)

        outLeft = shake_function_dir(left, leftAtt, *alphaBetaLeft)
        outRight = shake_function_dir(right, rightAtt, *alphaBetaRight)

        return outLeft, outRight

class shakeFunctionDir(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x1, x2, alpha, beta):
        
        ctx.save_for_backward(beta, alpha)
        y = x1 * alpha + x2 * (2.0 - alpha)
        
        return y

    @staticmethod
    def backward(ctx, grad_output):
        
        beta, _ = ctx.saved_variables
        g_x1 = g_x2 = g_alpha = g_beta = None

        if ctx.needs_input_grad[0]:
            g_x1 = grad_output * beta
        if ctx.needs_input_grad[1]:
            g_x2 = grad_output * (2.0 - beta)
       
        return g_x1, g_x2, g_alpha, g_beta

shake_function_dir = shakeFunctionDir.apply


def get_alpha_beta_dir(shake_config, is_cuda):
    
    forward_shake, backward_shake = shake_config

    if forward_shake:
        alpha = 2.0 * torch.rand(1)
    else:
        alpha = torch.FloatTensor([1.0])

    if backward_shake:
        beta = 2.0 * torch.rand(1)
    else:
        beta = torch.FloatTensor([1.0])
        
    if is_cuda:
        alpha = alpha.cuda()
        beta = beta.cuda()

    return alpha, beta

class sliceAttEncDecBlockShake(nn.Module):

    '''
    att_type:   
    three different attention types
    1. 0 denotes attention for both left and right
    2. 1 denotes attention for only left
    3. 2 denotes attention for only right
    '''

    def __init__(self, att_type = 0, shake_config = [False, False], with_ori = False):

        super(sliceAttEncDecBlockShake, self).__init__()

        self.att_type = att_type
        self.shake_config = shake_config
        self.with_ori = with_ori
        self.attDepth = sliceAttEncDecBlockDepth(att_type = self.att_type, with_ori = self.with_ori)
        self.attWidth = sliceAttEncDecBlockWidth(att_type = self.att_type, with_ori = self.with_ori)
        self.attHeight = sliceAttEncDecBlockHeight(att_type = self.att_type, with_ori = self.with_ori)

    def forward(self, left, right):
        
        leftDepth, rightDepth = self.attDepth(left, right)
        leftWidth, rightWidth = self.attWidth(left, right)
        leftHeight, rightHeight = self.attHeight(left, right)

        if self.training:
            shake_config = self.shake_config
        else:
            shake_config = [False, False]

        alphaBetaLeft = get_alpha_beta(shake_config, left.is_cuda)
        alphaBetaRight = get_alpha_beta(shake_config, right.is_cuda)

        outLeft = shake_function(left, leftDepth, leftWidth, leftHeight, *alphaBetaLeft)
        outRight = shake_function(right, rightDepth, rightWidth, rightHeight, *alphaBetaRight)

        return outLeft, outRight

class shakeFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x1, x2, x3, x4, 
        alpha1, alpha2, alpha3, alpha4, 
        beta1, beta2, beta3, beta4):
        
        ctx.save_for_backward(beta1, beta2, beta3, beta4)
        y = x1 * alpha1 + x2 * alpha2 + x3 * alpha3 + x4 * alpha4
        
        return y

    @staticmethod
    def backward(ctx, grad_output):
        
        beta1, beta2, beta3, beta4 = ctx.saved_variables
        g_x1 = g_x2 = g_x3 = g_x4 = None
        g_alpha1 = g_alpha2 = g_alpha3 = g_alpha4 = None 
        g_beta1 = g_beta2 = g_beta3 = g_beta4 = None 

        if ctx.needs_input_grad[0]:
            g_x1 = grad_output * beta1
        if ctx.needs_input_grad[1]:
            g_x2 = grad_output * beta2
        if ctx.needs_input_grad[2]:
            g_x3 = grad_output * beta3
        if ctx.needs_input_grad[3]:
            g_x4 = grad_output * beta4
        
        return g_x1, g_x2, g_x3, g_x4, g_alpha1, g_alpha2, g_alpha3, g_alpha4, g_beta1, g_beta2, g_beta3, g_beta4

shake_function = shakeFunction.apply


def get_alpha_beta(shake_config, is_cuda):
    
    forward_shake, backward_shake = shake_config

    if forward_shake:
        a = torch.rand(4)
        a = a / a.sum() * 4.0
        a1, a2, a3, a4 = a
    else:
        a1, a2, a3, a4 = torch.FloatTensor([1.0, 1.0, 1.0, 1.0])

    if backward_shake:
        b = torch.rand(4)
        b = b / b.sum() * 4.0
        b1, b2, b3, b4 = b
    else:
        b1, b2, b3, b4 = torch.FloatTensor([1.0, 1.0, 1.0, 1.0])

    if is_cuda:
        a1 = a1.cuda()
        a2 = a2.cuda()
        a3 = a3.cuda()
        a4 = a4.cuda()
        b1 = b1.cuda()
        b2 = b2.cuda()
        b3 = b3.cuda()
        b4 = b4.cuda()

    return a1, a2, a3, a4, b1, b2, b3, b4
    

class sliceSelfAttBlockND(nn.Module):

    def __init__(self):

        super(sliceSelfAttBlockND, self).__init__()
        
        self.beta = Parameter(torch.tensor(1.0))

    def forward(self, x):

        batch_size = x.size()[0]
        depth_size = x.size()[2]

        # n * c * d * h * w
        x_ori = x.permute(0, 2, 1, 3, 4)
        # n * d * c * h * w
        x_ = x_ori.contiguous().view(batch_size, depth_size, -1)
        # n * d * chw
        x_t = x_.permute(0, 2, 1)
        # n * chw * d

        attention = torch.matmul(x_, x_t)
        # n * d * d
        attention = F.softmax(attention, dim = -1)
        # n * d * d

        out1 = torch.matmul(attention, x_)
        out2 = self.beta * out1 + x_

        out2 = out2.view(batch_size, depth_size, *x_ori.size()[2:])        
        out2 = out2.permute(0, 2, 1, 3, 4)

        return out2

class channelAttentionBlockND(nn.Module):

    def __init__(self, in_channels, is_weighted):
        
        super(channelAttentionBlockND, self).__init__()
        
        self.in_channels = in_channels
        self.beta = Parameter(torch.tensor(1.0))
        self.is_weighted = is_weighted

    def forward(self, x):
        '''
        param x: (n, c, d, h, w)
        return: the same size as x
        '''

        batch_size = x.size()[0]
        channel_size = self.in_channels

        x_ = x.view(batch_size, channel_size, -1)
        # n * c * dhw
        x_t = x_.permute(0, 2, 1)
        # n * dhw * c
        attention = torch.matmul(x_, x_t)
        attention = F.softmax(attention, dim = -1)
        # n * c * c

        if self.is_weighted:

            attention = torch.sum(attention, dim = 1)
            # n * c
            attention = F.softmax(attention, dim = -1)
            # n * c            
            out1 = attention.unsqueeze(2) * x_
            # n * c * dhw  
        else: 
            out1 = torch.matmul(attention, x_)
            # n * c * dhw
            attention = torch.sum(attention, dim = 1)
            # n * c
            attention = F.softmax(attention, dim = -1)
            # n * c
        
        out2 = self.beta * out1 + x_
        out2 = out2.view(batch_size, channel_size, *x.size()[2:])        
            # n * c * d * h * w

        return out2, attention

class _nonLocalBlockND(nn.Module):
  
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=False, bn_layer=False):
        super(_nonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class nonLocalBlock1D(_nonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(nonLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class nonLocalBlock2D(_nonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(nonLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class nonLocalBlock3D(_nonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(nonLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)
