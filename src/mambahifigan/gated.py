"""Adaptive gated fusion -- the proposed architecture.

At each upsampling stage the generator holds a learnable scalar w_i. With alpha_i = sigmoid(w_i), the stage output is

    y = alpha_i * BiMamba(x) + (1 - alpha_i) * MRF(x)

w_i is initialised to 0, so training starts at an even split and the network moves the balance itself. `HifiganGeneratorHybrid1GatedFusion` is the configuration that produced the best feature-matching and L1 spectrogram losses in the study."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.lobes.models.HifiGAN import LRELU_SLOPE, ResBlock1, ResBlock2
from speechbrain.nnet.CNN import Conv1d, ConvTranspose1d

try:
    from mamba_ssm import Mamba2
except ImportError:  # the generators that need it raise on construction
    Mamba2 = None

from ._helpers import init_weights
from .blocks import BiMamba2Block


class HifiganGeneratorHybrid1GatedFusion(torch.nn.Module):
    """HiFiGAN Generator with Multi-Receptive Field Fusion (MRF)
    Hybrid 1 + Gated Fusion

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
        
        # --- NEW: Gating Parameters ---
        # Initialize with 0.0 so sigmoid(0.0) = 0.5 (equal weight to start)
        self.mixing_gates = nn.ParameterList()

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
            # Add a learnable scalar for this layer
            self.mixing_gates.append(nn.Parameter(torch.zeros(1)))

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
            
            # --- GATED FUSION START ---
            
            # 1. Calculate Convolutional Branch (Texture)
            conv_sum = None
            for j in range(self.num_kernels):
                res_out = self.resblocks[i * self.num_kernels + j](o)
                if conv_sum is None:
                    conv_sum = res_out
                else:
                    conv_sum += res_out
            
            # Average the Conv outputs
            conv_out = conv_sum / self.num_kernels

            # 2. Calculate Mamba Branch (Global Context)
            mamba_out = self.mambas[i](o)

            # 3. Dynamic Mixing
            # Calculate alpha (0 to 1)
            alpha = torch.sigmoid(self.mixing_gates[i])
            
            # Fuse: alpha controls how much we trust Mamba vs Conv
            # If alpha is high -> Trust Mamba more
            # If alpha is low  -> Trust Conv more
            o = (alpha * mamba_out) + ((1 - alpha) * conv_out)
            
            # --- GATED FUSION END ---

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


class HifiganGeneratorMambaMRF_Gated(torch.nn.Module):
    """
    HiFiGAN with Mamba-MRF Replacement AND Gated Fusion.
    
    Architecture:
    1. Replacement: Largest Conv Kernel (11) is REMOVED and REPLACED by Mamba.
    2. Gating: Mamba (Structure) and remaining Convs (Texture) are fused via a learnable gate.
    
    This combines the low L1 loss capability of Gating with the clean separation of Replacement.
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
        
        # --- KANAL KİLİDİ: 80 ---
        in_channels = 80 
        # ----------------------

        self.inference_padding = inference_padding
        self.num_upsamples = len(upsample_factors)
        
        # Replacement Mantığı: Son kernel'i atıyoruz.
        self.num_conv_kernels = len(resblock_kernel_sizes) - 1 
        
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2

        # 1. Initial Conv
        self.conv_pre = weight_norm(nn.Conv1d(
            in_channels=in_channels,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3
        ))
        
        # 2. Layers
        self.ups = nn.ModuleList()
        self.mambas = nn.ModuleList() 
        self.resblocks = nn.ModuleList()
        
        # --- GATING PARAMETRELERİ ---
        self.mixing_gates = nn.ParameterList()

        # ResBlock parametrelerini ayıkla (Sonuncuyu atıyoruz!)
        active_kernel_sizes = resblock_kernel_sizes[:self.num_conv_kernels]
        active_dilation_sizes = resblock_dilation_sizes[:self.num_conv_kernels]

        for i, (u, k) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            # Upsample
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(
                    in_channels=upsample_initial_channel // (2**i),
                    out_channels=upsample_initial_channel // (2**(i+1)),
                    kernel_size=k,
                    stride=u,
                    padding=(k-u)//2
                )
            ))
            
            ch = upsample_initial_channel // (2**(i+1))
            
            # A. Mamba Branch (Structure - Kernel 11'in yerine)
            self.mambas.append(BiMamba2Block(channels=ch, d_state=d_state))
            
            # B. Conv Branches (Texture - Kernel 3 ve 7)
            for j, (k_res, d_res) in enumerate(zip(active_kernel_sizes, active_dilation_sizes)):
                self.resblocks.append(resblock(ch, k_res, d_res))
            
            # C. Gate Parametresi (Her katman için 1 tane öğrenilebilir sayı)
            self.mixing_gates.append(nn.Parameter(torch.zeros(1)))

        # 3. Post Conv
        self.conv_post = weight_norm(nn.Conv1d(
            in_channels=ch, 
            out_channels=1, 
            kernel_size=7, 
            stride=1, 
            padding=3,
            bias=conv_post_bias
        ))
        
        if cond_channels > 0:
            self.cond_layer = weight_norm(nn.Conv1d(cond_channels, upsample_initial_channel, 1))

        self.apply(init_weights)

    def forward(self, x, g=None):
        o = self.conv_pre(x)
        if hasattr(self, "cond_layer"):
            o = o + self.cond_layer(g)

        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o) 
            
            # --- REPLACEMENT + GATED FUSION ---
            
            # 1. Conv Dalları (Texture)
            conv_sum = None
            for j in range(self.num_conv_kernels):
                res_out = self.resblocks[i*self.num_conv_kernels + j](o)
                if conv_sum is None:
                    conv_sum = res_out
                else:
                    conv_sum += res_out
            
            # Conv Ortalaması
            conv_out = conv_sum / self.num_conv_kernels
            
            # 2. Mamba Dalı (Structure)
            mamba_out = self.mambas[i](o)
            
            # 3. Gated Mixing
            # Alpha: Ne kadar Mamba kullanacağız?
            alpha = torch.sigmoid(self.mixing_gates[i])
            
            # Karışım: alpha * Mamba + (1-alpha) * Conv
            o = (alpha * mamba_out) + ((1 - alpha) * conv_out)
            # -------------------------------------
            
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

    def remove_weight_norm(self):
        print('Removing weight norm...')
        remove_weight_norm(self.conv_pre)
        for l in self.ups: remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()
        for l in self.mambas: 
            if hasattr(l, "remove_weight_norm"): l.remove_weight_norm()
        remove_weight_norm(self.conv_post)
        if hasattr(self, "cond_layer"): remove_weight_norm(self.cond_layer)

    @torch.no_grad()
    def inference(self, c, padding=True):
        if padding:
            c = torch.nn.functional.pad(c, (self.inference_padding, self.inference_padding), "replicate")
        return self.forward(c)
