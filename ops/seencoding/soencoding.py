##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Package Core NN Modules."""
import torch
import torch.nn as nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
# from ..MPNCOV.python import MPNCOV

from ..functions import scaled_l2, aggregate, pairwise_cosine

__all__ = ['SoEncoding', 'Vlad', 'EncodingDrop', 'Inspiration', 'UpsampleConv2d']



def covpool(x):
    # batchsize = x.data.shape[0]
    # dim = x.data.shape[1]
    M = x.data.shape[2]
    y = (1./M)*x.bmm(x.transpose(1, 2))
    # I = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchsize,1,1)
    # I_normX = I*1e-7
    # y = y + I_normX   
    return y

class Sqrtm_autograd(nn.Module):
    '''
    refer to Eq. (2-4)
    the same as Sqrtm implemented by autograd package of PyTorch
    the difference between Sqrtm and Sqrtm_autograd has been checked in check_sqrtm()
    '''
    def __init__(self, norm_type, num_iter):
        super(Sqrtm_autograd, self).__init__()
        self.norm_type = norm_type
        self.num_iter = num_iter

    def forward(self, A):
        dtype = A.dtype
        batchSize = A.data.shape[0]
        dim = A.data.shape[1]
        normA = []
        traces = []
        # pre normalization
        if self.norm_type == 'AF':
            normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
            Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
        elif self.norm_type == 'AT':   # AT: trace
            diags = []
            for i in range(batchSize):
                diags.append(torch.unsqueeze(torch.diag(A[i, :, :]), dim=0))
            diags = torch.cat(diags)
            traces = torch.unsqueeze(torch.sum(diags, dim=-1, keepdim=True), dim=-1)  # nx1x1

            # I3 = 3.0 * torch.eye(dim, dim, requires_grad=False, device=A.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
            # traces = (1.0 / 3.0) * A.mul(I3).sum(dim=1).sum(dim=1).unsqueeze(1).unsqueeze(1)  # nx1x1

            I = 1e-12 * torch.ones(batchSize, 1, 1, device=traces.device)  # 1e-10, 1e-12
            traces += I
            Y = A.div(traces.expand_as(A))
        
        else:
            raise NameError('invalid normalize type {}'.format(self.norm_type))

        # Iteration
        I = Variable(torch.eye(dim, dim, device=A.device).view(1, dim, dim).
                     repeat(batchSize, 1, 1).type(dtype), requires_grad=False)
        Z = Variable(torch.eye(dim, dim, device=A.device).view(1, dim, dim).
                     repeat(batchSize, 1, 1).type(dtype), requires_grad=False)

        for i in range(self.num_iter):
            T = 0.5 * (3.0 * I - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)

        # post normalization
        if self.norm_type == 'AF':
            sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
        else:
            sA = Y * torch.sqrt(traces).expand_as(A)            
        del I, Z
        return sA


def sqrtm(x, numIters):
    batchSize = x.data.shape[0]
    dim = x.data.shape[1]
    dtype = x.dtype
    I = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
    normA = x.mul(I).sum(dim=1).sum(dim=1)
    Y = x.div(normA.view(batchSize, 1, 1).expand_as(x))
    Z = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    y = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
    return y

class SoEncoding(Module):
    r"""
    Encoding Layer: a learnable residual encoder.

    .. image:: _static/img/cvpr17.svg
        :width: 50%
        :align: center

    Encoding Layer accpets 3D or 4D inputs.
    It considers an input featuremaps with the shape of :math:`C\times H\times W`
    as a set of C-dimentional input features :math:`X=\{x_1, ...x_N\}`, where N is total number
    of features given by :math:`H\times W`, which learns an inherent codebook
    :math:`D=\{d_1,...d_K\}` and a set of smoothing factor of visual centers
    :math:`S=\{s_1,...s_K\}`. Encoding Layer outputs the residuals with soft-assignment weights
    :math:`e_k=\sum_{i=1}^Ne_{ik}`, where

    .. math::

        e_{ik} = \frac{exp(-s_k\|r_{ik}\|^2)}{\sum_{j=1}^K exp(-s_j\|r_{ij}\|^2)} r_{ik}

    and the residuals are given by :math:`r_{ik} = x_i - d_k`. The output encoders are
    :math:`E=\{e_1,...e_K\}`.

    Args:
        D: dimention of the features or feature channels
        K: number of codeswords

    Shape:
        - Input: :math:`X\in\mathcal{R}^{B\times N\times D}` or
          :math:`\mathcal{R}^{B\times D\times H\times W}` (where :math:`B` is batch,
          :math:`N` is total number of features or :math:`H\times W`.)
        - Output: :math:`E\in\mathcal{R}^{B\times K\times D}`

    Attributes:
        codewords (Tensor): the learnable codewords of shape (:math:`K\times D`)
        scale (Tensor): the learnable scale factor of visual centers

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

        Hang Zhang, Jia Xue, and Kristin Dana. "Deep TEN: Texture Encoding Network."
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2017*

    Examples:
        >>> import encoding
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from torch.autograd import Variable
        >>> B,C,H,W,K = 2,3,4,5,6
        >>> X = Variable(torch.cuda.DoubleTensor(B,C,H,W).uniform_(-0.5,0.5), requires_grad=True)
        >>> layer = encoding.Encoding(C,K).double().cuda()
        >>> E = layer(X)
    """
    def __init__(self, D, K):
        super(SoEncoding, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = Parameter(torch.Tensor(K), requires_grad=True)
        # self.scale = Parameter(torch.Tensor(K), requires_grad=False)
        self.reset_params()

    def reset_params(self):
        std1 = 1./((self.K*self.D)**(1/2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)
        # self.scale.data.uniform_(-1, -1)

    def forward(self, X):
        # input X is a 4D tensor
        # assert(X.size(1) == self.D)
        B, D = X.size(0), self.D
        if X.dim() == 3:
            # BxDxN => BxNxD
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW => Bx(HW)xD
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        # assignment weights BxNxK
        s_l2 = scaled_l2(X, self.codewords, self.scale)  # bsx3600x32 (bsxNxc) sl_{ik} = s_k \|x_i-c_k\|^2  added by sql
        A = F.softmax(s_l2, dim=2)  # bsx3600x32 (bsxNxc)  added by sql
        # A = F.softmax(scaled_l2(X, self.codewords, self.scale), dim=2)  # softmax is applied to the dimension along K

        # calculate residuals to each clusters
        if self.K == 1:
            residual = X.expand(self.codewords.size(0), -1, -1, -1).permute(1, 2, 0, 3) - \
                    self.codewords.unsqueeze(0).unsqueeze(0)  # for k=1
        else:
            residual = X.expand(self.codewords.size(0), -1, -1, -1).permute(1, 2, 0, 3).contiguous()  # for k>1
        residual *= A.unsqueeze(3)

        cov = torch.zeros(B, self.K, X.size(2), X.size(2), requires_grad=True).cuda()
        # cov = torch.zeros(B, self.K, X.size(2), X.size(2), requires_grad=True, device=X.device)
        sqrt_auto = Sqrtm_autograd(norm_type='AT', num_iter=5)
        for codeword in range(self.K):
            feature = residual[:, :, codeword, :]
            feature = covpool(feature.transpose(1, 2))
            # feature = sqrtm(feature, 5)  # 3G-Net
            feature = sqrt_auto(feature)  #
            cov[:, codeword] = feature


            # feature = covpool(feature.transpose(1, 2))
            # feature = sqrtm(feature, 5)

            # feature = feature.transpose(1, 2).bmm(feature)
            # cov[:, codeword] = torch.div(feature, X.size(1))

            # cov[:, codeword] = sqrtm(torch.div(feature, X.size(1)), 5)
            # cov[:, codeword] = MPNCOV.SqrtmLayer(torch.div(feature, X.size(1)), 5)

        # # sum aggregate
        # sum_cov = cov.sum(dim=1)  # bsxDxD
        # # triu part of cov
        # dim = int(sum_cov.size(1)*(sum_cov.size(1)+1)/2)
        # sum_cov = MPNCOV.TriuvecLayer(sum_cov).view(B, dim, 1, 1)
        # return sum_cov
        return cov

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'KxCxC' + '=>' + str(self.K)  + 'x' \
            + str(self.D) + 'x' + str(self.D) + ')'

#########################################################################################

class Vlad(Module):
    def __init__(self, D, K):
        super(Vlad, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1./((self.K*self.D)**(1/2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        # input X is a 4D tensor
        assert(X.size(1) == self.D)
        B, D = X.size(0), self.D
        if X.dim() == 3:
            # BxDxN => BxNxD
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW => Bx(HW)xD
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Vlad Layer unknown input dims!')
        # assignment weights BxNxK
        # s_l2 = scaled_l2(X, self.codewords, self.scale)  # sl_{ik} = s_k \|x_i-c_k\|^2  added by sql
        # A = F.softmax(s_l2, dim=2)  # added by sql
        A = F.softmax(scaled_l2(X, self.codewords, self.scale), dim=2)  # softmax is applied to the dimension along K
        # aggregate
        E = aggregate(A, X, self.codewords)  # e_{k} = \sum_{i=1}^{N} a_{ik} (x_i - d_k)
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.D) + '=>' + str(self.K) + 'x' \
            + str(self.D) + ')'
#########################################################################################


class EncodingDrop(Module):
    r"""Dropout regularized Encoding Layer.
    """
    def __init__(self, D, K):
        super(EncodingDrop, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1./((self.K*self.D)**(1/2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def _drop(self):
        if self.training:
            self.scale.data.uniform_(-1, 0)
        else:
            self.scale.data.zero_().add_(-0.5)

    def forward(self, X):
        # input X is a 4D tensor
        assert(X.size(1) == self.D)
        if X.dim() == 3:
            # BxDxN
            B, D = X.size(0), self.D
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW
            B, D = X.size(0), self.D
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        self._drop()
        # assignment weights
        A = F.softmax(scaled_l2(X, self.codewords, self.scale), dim=2)
        # aggregate
        E = aggregate(A, X, self.codewords)
        self._drop()
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.D) + '=>' + str(self.K) + 'x' \
            + str(self.D) + ')'


class Inspiration(Module):
    r"""
    Inspiration Layer (CoMatch Layer) enables the multi-style transfer in feed-forward
    network, which learns to match the target feature statistics during the training.
    This module is differentialble and can be inserted in standard feed-forward network
    to be learned directly from the loss function without additional supervision.

    .. math::
        Y = \phi^{-1}[\phi(\mathcal{F}^T)W\mathcal{G}]

    Please see the `example of MSG-Net <./experiments/style.html>`_
    training multi-style generative network for real-time transfer.

    Reference:
        Hang Zhang and Kristin Dana. "Multi-style Generative Network for Real-time Transfer."
        *arXiv preprint arXiv:1703.06953 (2017)*
    """
    def __init__(self, C, B=1):
        super(Inspiration, self).__init__()
        # B is equal to 1 or input mini_batch
        self.weight = Parameter(torch.Tensor(1, C, C), requires_grad=True)
        # non-parameter buffer
        self.G = Variable(torch.Tensor(B, C, C), requires_grad=True)
        self.C = C
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(0.0, 0.02)

    def setTarget(self, target):
        self.G = target

    def forward(self, X):
        # input X is a 3D feature map
        self.P = torch.bmm(self.weight.expand_as(self.G), self.G)
        return torch.bmm(self.P.transpose(1, 2).expand(X.size(0), self.C, self.C),
                         X.view(X.size(0), X.size(1), -1)).view_as(X)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.C) + ')'


class UpsampleConv2d(Module):
    r"""
    To avoid the checkerboard artifacts of standard Fractionally-strided Convolution,
    we adapt an integer stride convolution but producing a :math:`2\times 2` outputs for
    each convolutional window.

    .. image:: _static/img/upconv.png
        :width: 50%
        :align: center

    Reference:
        Hang Zhang and Kristin Dana. "Multi-style Generative Network for Real-time Transfer."
        *arXiv preprint arXiv:1703.06953 (2017)*

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        output_padding (int or tuple, optional): Zero-padding added to one side of the output.
          Default: 0
        groups (int, optional): Number of blocked connections from input channels to output
          channels. Default: 1
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        scale_factor (int): scaling factor for upsampling convolution. Default: 1

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          :math:`H_{out} = scale * (H_{in} - 1) * stride[0] - 2 * padding[0] + kernel\_size[0] + output\_padding[0]`
          :math:`W_{out} = scale * (W_{in} - 1) * stride[1] - 2 * padding[1] + kernel\_size[1] + output\_padding[1]`

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (in_channels, scale * scale * out_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (scale * scale * out_channels)

    Examples:
        >>> # With square kernels and equal stride
        >>> m = nn.UpsampleCov2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.UpsampleCov2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 100))
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> input = autograd.Variable(torch.randn(1, 16, 12, 12))
        >>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nn.UpsampleCov2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, scale_factor=1,
                 bias=True):
        super(UpsampleConv2d, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.scale_factor = scale_factor
        self.weight = Parameter(torch.Tensor(
            out_channels * scale_factor * scale_factor,
            in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(
                out_channels * scale_factor * scale_factor))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        out = F.conv2d(input, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return F.pixel_shuffle(out, self.scale_factor)
