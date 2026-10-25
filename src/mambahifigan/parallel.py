"""Parallel CNN + Mamba branches with fixed (non-learnable) fusion, and the Mamba-only ablation.

These are the configurations the gated design is measured against: they put the same Bidirectional Mamba branch beside the MRF module but combine the two with a constant weight instead of a learnable one."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.lobes.models.HifiGAN import LRELU_SLOPE, ResBlock1, ResBlock2
from speechbrain.nnet.CNN import Conv1d, ConvTranspose1d

try:
    from mamba_ssm import Mamba2
except ImportError:  # the generators that need it raise on construction
    Mamba2 = None

from .blocks import BiMamba2Block


class HifiganGeneratorHybrid1Summation(torch.nn.Module):
    """HiFiGAN Generator with Multi-Receptive Field Fusion (MRF)
    hybrid1

    Arguments
    ---------
    in_channels : int
        number of input tensor channels.
    out_channels : int
        number of output tensor channels.
    resblock_type : str
        type of the `ResBlock`. '1' or '2'.
    resblock_dilation_sizes : List[List[int]]
        list of dilation values in each layer of a `ResBlock`.
    resblock_kernel_sizes : List[int]
        list of kernel sizes for each `ResBlock`.
    upsample_kernel_sizes : List[int]
        list of kernel sizes for each transposed convolution.
    upsample_initial_channel : int
        number of channels for the first upsampling layer. This is divided by 2
        for each consecutive upsampling layer.
    upsample_factors : List[int]
        upsampling factors (stride) for each upsampling layer.
    inference_padding : int
       constant padding applied to the input at inference time. Defaults to 5.
    cond_channels : int
        If provided, adds a conv layer to the beginning of the forward.
    conv_post_bias : bool
        Whether to add a bias term to the final conv.
    d_state : int
        SSM state dimension of the Mamba blocks. Defaults to 128, the value hardcoded
        in the file as this configuration was run.

    Example
    -------
    >>> inp_tensor = torch.rand([4, 80, 33])
    >>> hifigan_generator = HifiganGenerator(
    ...     in_channels=80,
    ...     out_channels=1,
    ...     resblock_type="1",
    ...     resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ...     resblock_kernel_sizes=[3, 7, 11],
    ...     upsample_kernel_sizes=[16, 16, 4, 4],
    ...     upsample_initial_channel=512,
    ...     upsample_factors=[8, 8, 2, 2],
    ... )
    >>> out_tensor = hifigan_generator(inp_tensor)
    >>> out_tensor.shape
    torch.Size([4, 1, 8448])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        resblock_type,
        resblock_dilation_sizes,
        resblock_kernel_sizes,
        upsample_kernel_sizes,
        upsample_initial_channel,
        upsample_factors,
        inference_padding=5,
        cond_channels=0,
        conv_post_bias=True,
        d_state=128,
    ):
        super().__init__()
        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        # initial upsampling layers
        self.conv_pre = Conv1d(
            in_channels=in_channels,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding="same",
            skip_transpose=True,
            weight_norm=True,
        )
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2
        # upsampling layers
        self.ups = nn.ModuleList()
        self.mambas = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(upsample_factors, upsample_kernel_sizes)
        ):
            self.ups.append(
                ConvTranspose1d(
                    in_channels=upsample_initial_channel // (2**i),
                    out_channels=upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2,
                    skip_transpose=True,
                    weight_norm=True,
                )
            )
        # MRF blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.mambas.append(BiMamba2Block(channels=ch, d_state=d_state))
            for _, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))
        # post convolution layer
        self.conv_post = Conv1d(
            in_channels=ch,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding="same",
            skip_transpose=True,
            bias=conv_post_bias,
            weight_norm=True,
        )
        if cond_channels > 0:
            self.cond_layer = Conv1d(
                in_channels=cond_channels,
                out_channels=upsample_initial_channel,
                kernel_size=1,
            )

    def forward(self, x, g=None):
        """
        Arguments
        ---------
        x : torch.Tensor (batch, channel, time)
            feature input tensor.
        g : torch.Tensor (batch, 1, time)
            global conditioning input tensor.

        Returns
        -------
        The generator outputs
        """

        o = self.conv_pre(x)
        if hasattr(self, "cond_layer"):
            o = o + self.cond_layer(g)
        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o)
            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i * self.num_kernels + j](o)
                else:
                    z_sum += self.resblocks[i * self.num_kernels + j](o)
            # Add Mamba branch
            z_sum += self.mambas[i](o)
            o = z_sum / (self.num_kernels + 1)
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

    def remove_weight_norm(self):
        """This functions removes weight normalization during inference."""

        for layer in self.ups:
            layer.remove_weight_norm()
        for layer in self.resblocks:
            layer.remove_weight_norm()
        for layer in self.mambas:
            if hasattr(layer, "remove_weight_norm"):
                layer.remove_weight_norm()
        self.conv_pre.remove_weight_norm()
        self.conv_post.remove_weight_norm()

    @torch.no_grad()
    def inference(self, c, padding=True):
        """The inference function performs a padding and runs the forward method.

        Arguments
        ---------
        c : torch.Tensor (batch, channel, time)
            feature input tensor.
        padding : bool
            Whether to pad tensor before forward.

        Returns
        -------
        The generator outputs
        """
        if padding:
            c = torch.nn.functional.pad(
                c, (self.inference_padding, self.inference_padding), "replicate"
            )
        return self.forward(c)


class HybridHifiganGenerator2(torch.nn.Module):
    """HiFiGAN Generator with Multi-Receptive Field Fusion (MRF) and Mamba in parallel.
    This version explicitly adds a Mamba branch to the full MRF (3 residual blocks).

    Arguments
    ---------
    in_channels : int
        number of input tensor channels.
    out_channels : int
        number of output tensor channels.
    resblock_type : str
        type of the `ResBlock`. '1' or '2'.
    resblock_dilation_sizes : List[List[int]]
        list of dilation values in each layer of a `ResBlock`.
    resblock_kernel_sizes : List[int]
        list of kernel sizes for each `ResBlock`.
    upsample_kernel_sizes : List[int]
        list of kernel sizes for each transposed convolution.
    upsample_initial_channel : int
        number of channels for the first upsampling layer. This is divided by 2
        for each consecutive upsampling layer.
    upsample_factors : List[int]
        upsampling factors (stride) for each upsampling layer.
    inference_padding : int
       constant padding applied to the input at inference time. Defaults to 5.
    cond_channels : int
        If provided, adds a conv layer to the beginning of the forward.
    conv_post_bias : bool
        Whether to add a bias term to the final conv.
    d_state : int
        The state dimension for the Mamba blocks.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        resblock_type,
        resblock_dilation_sizes,
        resblock_kernel_sizes,
        upsample_kernel_sizes,
        upsample_initial_channel,
        upsample_factors,
        inference_padding=5,
        cond_channels=0,
        conv_post_bias=True,
        d_state=64,
    ):
        super().__init__()
        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        # initial upsampling layers
        self.conv_pre = Conv1d(
            in_channels=in_channels,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding="same",
            skip_transpose=True,
            weight_norm=True,
        )
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2
        # upsampling layers
        self.ups = nn.ModuleList()
        self.mambas = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(upsample_factors, upsample_kernel_sizes)
        ):
            self.ups.append(
                ConvTranspose1d(
                    in_channels=upsample_initial_channel // (2**i),
                    out_channels=upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2,
                    skip_transpose=True,
                    weight_norm=True,
                )
            )
        # MRF blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.mambas.append(BiMamba2Block(channels=ch, d_state=d_state))
            for _, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))
        # post convolution layer
        self.conv_post = Conv1d(
            in_channels=ch,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding="same",
            skip_transpose=True,
            bias=conv_post_bias,
            weight_norm=True,
        )
        if cond_channels > 0:
            self.cond_layer = Conv1d(
                in_channels=cond_channels,
                out_channels=upsample_initial_channel,
                kernel_size=1,
            )

    def forward(self, x, g=None):
        """
        Arguments
        ---------
        x : torch.Tensor (batch, channel, time)
            feature input tensor.
        g : torch.Tensor (batch, 1, time)
            global conditioning input tensor.

        Returns
        -------
        The generator outputs
        """

        o = self.conv_pre(x)
        if hasattr(self, "cond_layer"):
            o = o + self.cond_layer(g)
        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o)
            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i * self.num_kernels + j](o)
                else:
                    z_sum += self.resblocks[i * self.num_kernels + j](o)
            # Add Mamba branch
            z_sum += self.mambas[i](o)
            o = z_sum / (self.num_kernels + 1)
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

    def remove_weight_norm(self):
        """This functions removes weight normalization during inference."""

        for layer in self.ups:
            layer.remove_weight_norm()
        for layer in self.resblocks:
            layer.remove_weight_norm()
        for layer in self.mambas:
            if hasattr(layer, "remove_weight_norm"):
                layer.remove_weight_norm()
        self.conv_pre.remove_weight_norm()
        self.conv_post.remove_weight_norm()

    @torch.no_grad()
    def inference(self, c, padding=True):
        """The inference function performs a padding and runs the forward method.

        Arguments
        ---------
        c : torch.Tensor (batch, channel, time)
            feature input tensor.
        padding : bool
            Whether to pad tensor before forward.

        Returns
        -------
        The generator outputs
        """
        if padding:
            c = torch.nn.functional.pad(
                c, (self.inference_padding, self.inference_padding), "replicate"
            )
        return self.forward(c)


class MambaOnlyHifiganGenerator(torch.nn.Module):
    """HiFiGAN Generator with Mamba blocks replacing the MRF module.

    Arguments
    ---------
    in_channels : int
        number of input tensor channels.
    out_channels : int
        number of output tensor channels.
    resblock_type : str
        Unused, kept for backward compatibility with original HiFiGAN config.
    resblock_dilation_sizes : List[List[int]]
        Unused, kept for backward compatibility with original HiFiGAN config.
    resblock_kernel_sizes : List[int]
        Unused, kept for backward compatibility with original HiFiGAN config.
    upsample_kernel_sizes : List[int]
        list of kernel sizes for each transposed convolution.
    upsample_initial_channel : int
        number of channels for the first upsampling layer. This is divided by 2
        for each consecutive upsampling layer.
    upsample_factors : List[int]
        upsampling factors (stride) for each upsampling layer.
    inference_padding : int
       constant padding applied to the input at inference time. Defaults to 5.
    cond_channels : int
        If provided, adds a conv layer to the beginning of the forward.
    conv_post_bias : bool
        Whether to add a bias term to the final conv.
    d_state : int
        The state dimension for the Mamba blocks.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        resblock_type,  # unused, for YAML/API compatibility
        resblock_dilation_sizes,  # unused, for YAML/API compatibility
        resblock_kernel_sizes,  # unused, for YAML/API compatibility
        upsample_kernel_sizes,
        upsample_initial_channel,
        upsample_factors,
        inference_padding=5,
        cond_channels=0,
        conv_post_bias=True,
        d_state=64,
    ):
        super().__init__()
        self.inference_padding = inference_padding
        self.num_upsamples = len(upsample_factors)

        # Initial convolutional layer
        self.conv_pre = Conv1d(
            in_channels=in_channels,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding="same",
            skip_transpose=True,
            weight_norm=True,
        )

        # Upsampling and Mamba blocks
        self.ups = nn.ModuleList()
        self.mambas = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(upsample_factors, upsample_kernel_sizes)
        ):
            in_ch = upsample_initial_channel // (2**i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                ConvTranspose1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2,
                    skip_transpose=True,
                    weight_norm=True,
                )
            )
            self.mambas.append(BiMamba2Block(channels=out_ch, d_state=d_state))

        # Final convolutional layer
        self.conv_post = Conv1d(
            in_channels=out_ch,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding="same",
            skip_transpose=True,
            bias=conv_post_bias,
            weight_norm=True,
        )

        # Conditional layer
        if cond_channels > 0:
            self.cond_layer = Conv1d(
                in_channels=cond_channels,
                out_channels=upsample_initial_channel,
                kernel_size=1,
            )

    def forward(self, x, g=None):
        """
        Arguments
        ---------
        x : torch.Tensor (batch, channel, time)
            feature input tensor.
        g : torch.Tensor (batch, 1, time)
            global conditioning input tensor.

        Returns
        -------
        The generator outputs
        """
        o = self.conv_pre(x)
        if hasattr(self, "cond_layer"):
            o = o + self.cond_layer(g)

        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o)
            o = self.mambas[i](o)

        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

    def remove_weight_norm(self):
        """This functions removes weight normalization during inference."""
        for layer in self.ups:
            layer.remove_weight_norm()
        for layer in self.mambas:
            if hasattr(layer, "remove_weight_norm"):
                layer.remove_weight_norm()
        self.conv_pre.remove_weight_norm()
        self.conv_post.remove_weight_norm()

    @torch.no_grad()
    def inference(self, c, padding=True):
        """The inference function performs a padding and runs the forward method."""
        if padding:
            c = torch.nn.functional.pad(
                c, (self.inference_padding, self.inference_padding), "replicate"
            )
        return self.forward(c)
