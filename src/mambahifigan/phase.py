"""Failure family 1: phase coherence.

Each of these modifies the waveform after it has been generated, or mixes streams produced at different sample rates. Audio is oscillatory and zero-centred, so both break phase alignment: the post-net learns to predict near-zero to minimise L1 and the output goes muffled; the dual-path sum produces destructive interference between two branches with different receptive fields; the multi-rate design adds aliasing on the way down and smeared transients on the way back up."""

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
from .blocks import LowResMambaBlock, MambaRefiner, MambaSideNetwork


class HifiganGeneratorWithPostNet(torch.nn.Module):
    """
    HiFiGAN with Mamba Post-Net Refiner.
    
    Architecture:
    1. Generator: Standard HiFi-GAN (No Mamba inside). Produces raw audio.
    2. Refiner: A separate Mamba network that takes the raw audio and 'polishes' it.
    
    Benefit:
    - Generator focuses on Texture (FM Loss).
    - Refiner focuses on Structure/Consistency (L1 Loss).
    - Decoupled optimization prevents them from ruining each other's work.
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
    ):
        super().__init__()
        
        # --- KANAL KİLİDİ ---
        in_channels = 80 
        # --------------------

        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2

        # --- PART 1: STANDARD HiFi-GAN GENERATOR ---
        self.conv_pre = weight_norm(nn.Conv1d(
            in_channels=in_channels,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3
        ))
        
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
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
            
            for j, (k_res, d_res) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k_res, d_res))

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

        # --- PART 2: MAMBA POST-NET REFINER ---
        self.refiner = MambaRefiner(d_model=64)
        # --------------------------------------

        self.apply(init_weights)

    def forward(self, x, g=None):
        # 1. Generate Coarse Audio (HiFi-GAN)
        o = self.conv_pre(x)
        if hasattr(self, "cond_layer"):
            o = o + self.cond_layer(g)

        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o)
            
            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i*self.num_kernels+j](o)
                else:
                    z_sum += self.resblocks[i*self.num_kernels+j](o)
            o = z_sum / self.num_kernels
            
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        coarse_audio = torch.tanh(o) # [Batch, 1, Time]
        
        # 2. Refine Audio (Mamba)
        # Ham sesi al, düzeltilmiş halini ver
        refined_audio = self.refiner(coarse_audio)
        
        # Değerlerin -1 ile 1 arasında kalmasını garantiye al
        return torch.tanh(refined_audio)

    def remove_weight_norm(self):
        print('Removing weight norm...')
        remove_weight_norm(self.conv_pre)
        for l in self.ups: remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()
        remove_weight_norm(self.conv_post)
        if hasattr(self, "cond_layer"): remove_weight_norm(self.cond_layer)
        
        # Refiner temizliği
        self.refiner.remove_weight_norm()

    @torch.no_grad()
    def inference(self, c, padding=True):
        if padding:
            c = torch.nn.functional.pad(c, (self.inference_padding, self.inference_padding), "replicate")
        return self.forward(c)


class HifiganGeneratorDualPath(torch.nn.Module):
    """
    HiFi-GAN Dual-Path Generator.
    
    Path 1 (Main): Standard HiFi-GAN (Conv). Handles Texture perfectly.
    Path 2 (Side): Mamba Network. Handles Structure/Correction.
    
    Fusion: y_final = y_conv + y_mamba
    
    Safety: Path 2 is Zero-Initialized. Training starts exactly at Baseline performance
    and improves from there. No risk of degradation.
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
    ):
        super().__init__()
        
        # --- KANAL KİLİDİ: 80 ---
        in_channels = 80 
        # ----------------------

        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2

        # --- PATH 1: STANDARD HiFi-GAN (DOKUNULMAMIŞ) ---
        self.conv_pre = weight_norm(nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3))
        
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2**(i+1)),
                    kernel_size=k,
                    stride=u,
                    padding=(k-u)//2
                )
            ))
            ch = upsample_initial_channel // (2**(i+1))
            for j, (k_res, d_res) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k_res, d_res))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3, bias=conv_post_bias))
        
        if cond_channels > 0:
            self.cond_layer = weight_norm(nn.Conv1d(cond_channels, upsample_initial_channel, 1))
            
        # --- PATH 2: MAMBA SIDE NETWORK (CO-PILOT) ---
        # Sadece Mamba'dan oluşan paralel bir ağ.
        # HiFi-GAN ile aynı upsample oranlarını kullanır.
        self.mamba_side_net = MambaSideNetwork(
            in_channels=in_channels,
            upsample_factors=upsample_factors,
            upsample_initial_channel=upsample_initial_channel // 2 # Daha hafif olsun diye yarı kanal
        )
        # ---------------------------------------------

        self.apply(init_weights)

    def forward(self, x, g=None):
        # --- PATH 1: CONV (TEXTURE) ---
        o = self.conv_pre(x)
        if hasattr(self, "cond_layer"):
            o = o + self.cond_layer(g)

        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o)
            z_sum = None
            for j in range(self.num_kernels):
                res_out = self.resblocks[i*self.num_kernels+j](o)
                if z_sum is None: z_sum = res_out
                else: z_sum += res_out
            o = z_sum / self.num_kernels
            
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        audio_conv = torch.tanh(o) # Orijinal HiFi-GAN çıktısı
        
        # --- PATH 2: MAMBA (CORRECTION) ---
        # Mamba ağından bağımsız bir düzeltme sinyali alıyoruz
        audio_correction = self.mamba_side_net(x)
        
        # --- FUSION ---
        # İki sinyali topluyoruz. 
        # Başlangıçta audio_correction = 0 olduğu için eğitim Baseline kalitesinde başlar.
        # Zamanla Mamba "yardım etmeyi" öğrenir.
        final_audio = audio_conv + audio_correction
        
        # Değerlerin -1, 1 aralığında kalmasını sağla (Safety clamp)
        return torch.clamp(final_audio, -1.0, 1.0)

    def remove_weight_norm(self):
        print('Removing weight norm...')
        remove_weight_norm(self.conv_pre)
        for l in self.ups: remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()
        remove_weight_norm(self.conv_post)
        if hasattr(self, "cond_layer"): remove_weight_norm(self.cond_layer)
        
        # Yan ağın normlarını da kaldır
        self.mamba_side_net.remove_weight_norm()

    @torch.no_grad()
    def inference(self, c, padding=True):
        if padding:
            c = torch.nn.functional.pad(c, (self.inference_padding, self.inference_padding), "replicate")
        return self.forward(c)


class HifiganGeneratorMultiRate(torch.nn.Module):
    """
    HiFiGAN with Multi-Rate Mamba (Low-Pass Guidance).
    Includes Memory-Safe Projections.
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
    ):
        super().__init__()
        
        # --- KANAL KİLİDİ ---
        in_channels = 80 
        # --------------------

        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        
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
        self.resblocks = nn.ModuleList()
        self.mamba_blocks = nn.ModuleList() 

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
            
            current_ch = upsample_initial_channel // (2**(i+1))
            
            # MRF (Conv - Texture)
            for j, (k_res, d_res) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(current_ch, k_res, d_res))
            
            # --- MULTI-RATE MAMBA (Structure) ---
            # scale_factor=2 ile sinyali yarı boyutta işler (Detayları görmez/bozmaz)
            self.mamba_blocks.append(LowResMambaBlock(channels=current_ch, scale_factor=2))

        # 3. Post Conv
        self.conv_post = weight_norm(nn.Conv1d(
            in_channels=current_ch, 
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
            o = self.ups[i](o) # Upsample
            
            # 1. MRF İşlemi (Conv)
            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i*self.num_kernels+j](o)
                else:
                    z_sum += self.resblocks[i*self.num_kernels+j](o)
            
            o = z_sum / self.num_kernels
            
            # 2. Mamba Correction
            # Low-Res ve Projeksiyonlu (Güvenli) Mamba
            o = self.mamba_blocks[i](o)
            
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

    def remove_weight_norm(self):
        print('Removing weight norm...')
        remove_weight_norm(self.conv_pre)
        for l in self.ups: remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()
        for l in self.mamba_blocks: l.remove_weight_norm()
            
        remove_weight_norm(self.conv_post)
        if hasattr(self, "cond_layer"): remove_weight_norm(self.cond_layer)

    @torch.no_grad()
    def inference(self, c, padding=True):
        if padding:
            c = torch.nn.functional.pad(c, (self.inference_padding, self.inference_padding), "replicate")
        return self.forward(c)
