"""Resolution-specific placement: Mamba at some stages only.

If global context helps, does it have to be everywhere? These apply the SSM branch only in the early low-resolution stages, only in the late high-resolution stages, or serially within a stage. All three land between the baseline and the gated design, which is what motivated applying the gated branch at every stage and letting alpha decide."""

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


class HifiganGeneratorBottleneck(torch.nn.Module):
    """
    HiFiGAN with Bottleneck Mamba Strategy.
    
    Logic:
    Mamba is applied ONLY in the low-resolution upsampling stages (Bottleneck).
    High-resolution stages use only Convolution to preserve texture/fidelity.
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
        
        # --- ACİL DURUM YAMASI ---
        # Veri setinden 80 kanal geldiği kesinleşti.
        # YAML ne derse desin, burayı 80'e kilitliyoruz.
        in_channels = 80 
        # -------------------------

        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        
        # Mamba'nın çalışacağı sınır (Toplam katmanların yarısı)
        # Örn: 4 katman varsa, sadece ilk 2'sinde (index 0 ve 1) Mamba çalışır.
        self.mamba_cutoff_index = self.num_upsamples // 2
        
        # ResBlock Tipi
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2

        # 1. Initial Convolution (Saf PyTorch)
        self.conv_pre = weight_norm(nn.Conv1d(
            in_channels=in_channels,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3
        ))
        
        # 2. Upsampling Layers
        self.ups = nn.ModuleList()
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

        # 3. Mamba ve MRF Blokları
        self.mambas = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i+1))
            
            # --- STRATEJİ: Mamba Sadece İlk Yarısında Var ---
            if i < self.mamba_cutoff_index:
                # Sadece cutoff öncesi katmanlar için Mamba oluşturuyoruz
                self.mambas.append(BiMamba2Block(channels=ch, d_state=d_state))
            
            # MRF (Conv) her zaman var
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        # 4. Post Convolution
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

        # Ağırlıkları başlat
        self.apply(init_weights)

    def forward(self, x, g=None):
        # x shape: [Batch, 80, Time]
        o = self.conv_pre(x)
        
        if hasattr(self, "cond_layer"):
            o = o + self.cond_layer(g)

        # Mamba listesi için ayrı bir sayaç tutuyoruz
        mamba_idx = 0

        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            
            # Upsample
            o = self.ups[i](o)
            
            # --- BOTTLENECK STRATEJİSİ ---
            # Sadece belirlenen index'ten küçükse Mamba uygula
            if i < self.mamba_cutoff_index:
                mamba_block = self.mambas[mamba_idx]
                mamba_out = mamba_block(o)
                o = o + mamba_out # Residual Bağlantı
                mamba_idx += 1
            # -----------------------------
            
            # MRF (Conv) Texture İşleme
            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i*self.num_kernels+j](o)
                else:
                    z_sum += self.resblocks[i*self.num_kernels+j](o)
            
            o = z_sum / self.num_kernels
            
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

    def remove_weight_norm(self):
        print('Removing weight norm...')
        remove_weight_norm(self.conv_pre)
        for l in self.ups: remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()
        
        # Mamba'da weight norm yok ama fonksiyonu çağırmak gerekebilir
        for l in self.mambas: 
            if hasattr(l, "remove_weight_norm"):
                l.remove_weight_norm()
            
        remove_weight_norm(self.conv_post)
        if hasattr(self, "cond_layer"): remove_weight_norm(self.cond_layer)

    @torch.no_grad()
    def inference(self, c, padding=True):
        if padding:
            c = torch.nn.functional.pad(c, (self.inference_padding, self.inference_padding), "replicate")
        return self.forward(c)


class HifiganGeneratorLateMamba(torch.nn.Module):
    """
    HiFiGAN with Late-Stage Mamba Strategy.
    
    Logic:
    Mamba is applied ONLY in the FINAL upsampling stages (High Resolution).
    Hypothesis: This helps correct Phase consistency in the raw waveform domain.
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
        
        # --- ACİL DURUM: Kanal Kilidi (Veri setine göre 80) ---
        in_channels = 80 
        # ------------------------------------------------------

        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        
        # STRATEJİ: Mamba ne zaman devreye girsin?
        # Sadece son 2 blokta çalışsın (En yüksek çözünürlük)
        # Örn: 4 katman varsa, index 2 ve 3'te çalışır.
        self.mamba_start_index = self.num_upsamples - 2
        
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2

        # 1. Initial Conv
        self.conv_pre = weight_norm(nn.Conv1d(
            in_channels=in_channels,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3
        ))
        
        # 2. Upsampling Layers
        self.ups = nn.ModuleList()
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

        # 3. Mamba ve MRF Blokları
        self.mambas = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i+1))
            
            # --- STRATEJİ DEĞİŞİKLİĞİ ---
            # Sadece SON bloklarsa Mamba ekle
            if i >= self.mamba_start_index:
                # Yüksek çözünürlükte hafıza patlamasın diye d_state küçültülebilir
                # Ama şimdilik 64 ile deneyelim (Eski 128 idi)
                self.mambas.append(BiMamba2Block(channels=ch, d_state=d_state))
            
            # MRF her zaman var
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        # 4. Post Conv
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

        mamba_idx = 0

        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            
            # Upsample
            o = self.ups[i](o)
            
            # --- LATE STAGE MAMBA STRATEJİSİ ---
            # Eğer son bloklardaysak Mamba'yı çalıştır
            if i >= self.mamba_start_index:
                mamba_block = self.mambas[mamba_idx]
                
                # Mamba'yı Residual olarak ekle
                mamba_out = mamba_block(o)
                o = o + mamba_out 
                
                mamba_idx += 1
            # -----------------------------------
            
            # MRF (Conv) Texture İşleme
            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i*self.num_kernels+j](o)
                else:
                    z_sum += self.resblocks[i*self.num_kernels+j](o)
            
            o = z_sum / self.num_kernels
            
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

    def remove_weight_norm(self):
        print('Removing weight norm...')
        remove_weight_norm(self.conv_pre)
        for l in self.ups: remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()
        
        # Mamba temizliği
        for l in self.mambas: 
            if hasattr(l, "remove_weight_norm"):
                l.remove_weight_norm()
            
        remove_weight_norm(self.conv_post)
        if hasattr(self, "cond_layer"): remove_weight_norm(self.cond_layer)

    @torch.no_grad()
    def inference(self, c, padding=True):
        if padding:
            c = torch.nn.functional.pad(c, (self.inference_padding, self.inference_padding), "replicate")
        return self.forward(c)


class HifiganGeneratorSerial(torch.nn.Module):
    """
    HiFiGAN Generator with Serial Mamba-Guided Refinement.
    Re-written using pure torch.nn layers to avoid SpeechBrain wrapper conflicts.
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
        
        # ResBlock Tipini Belirle
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2

        # ---------------------------------------------------------
        # KATMAN 1: Initial Convolution (Saf PyTorch)
        # ---------------------------------------------------------
        # SpeechBrain'in Conv1d'si yerine nn.Conv1d kullanıyoruz.
        # weight_norm'u manuel olarak sarıyoruz.
        self.conv_pre = weight_norm(nn.Conv1d(
            in_channels=in_channels,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3  # (kernel_size - 1) / 2 -> (7-1)/2 = 3 (Same padding)
        ))
        
        # ---------------------------------------------------------
        # KATMAN 2: Upsampling Layers (Saf PyTorch)
        # ---------------------------------------------------------
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            # Padding hesabı: (kernel - stride) // 2
            p = (k - u) // 2
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(
                    in_channels=upsample_initial_channel // (2**i),
                    out_channels=upsample_initial_channel // (2**(i+1)),
                    kernel_size=k,
                    stride=u,
                    padding=p
                )
            ))

        # ---------------------------------------------------------
        # KATMAN 3: Serial Modules (Mamba + ResBlocks)
        # ---------------------------------------------------------
        self.mambas = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i+1))
            
            # Mamba
            self.mambas.append(BiMamba2Block(channels=ch, d_state=d_state))
            
            # ResBlocks (SpeechBrain'den geldiği için weight_norm içindedir, dokunmuyoruz)
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        # ---------------------------------------------------------
        # KATMAN 4: Post Convolution (Saf PyTorch)
        # ---------------------------------------------------------
        self.conv_post = weight_norm(nn.Conv1d(
            in_channels=ch, 
            out_channels=1, 
            kernel_size=7, 
            stride=1, 
            padding=3,
            bias=conv_post_bias
        ))
        
        # Conditional Layer
        if cond_channels > 0:
            self.cond_layer = weight_norm(nn.Conv1d(
                in_channels=cond_channels, 
                out_channels=upsample_initial_channel, 
                kernel_size=1
            ))

        # Ağırlıkları Başlat
        self.apply(init_weights)

    def forward(self, x, g=None):
        
        # 1. Initial Conv
        o = self.conv_pre(x)
        
        if hasattr(self, "cond_layer"):
            o = o + self.cond_layer(g)

        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            
            # 2. Upsample
            o = self.ups[i](o)
            
            # --- SERIAL REFINEMENT ---
            
            # 3. Mamba Context (Guidance)
            mamba_context = self.mambas[i](o)
            o_guided = o + mamba_context
            
            # 4. Conv MRF (Texture)
            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i*self.num_kernels+j](o_guided)
                else:
                    z_sum += self.resblocks[i*self.num_kernels+j](o_guided)
            
            o = z_sum / self.num_kernels
            
        # 5. Final Post-Processing
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

    def remove_weight_norm(self):
        print('Removing weight norm...')
        # PyTorch weight_norm kaldırma
        remove_weight_norm(self.conv_pre)
        for l in self.ups: remove_weight_norm(l)
        
        # ResBlocks (SpeechBrain methodu)
        for l in self.resblocks: l.remove_weight_norm()
        
        # Mamba
        for l in self.mambas: l.remove_weight_norm()
            
        remove_weight_norm(self.conv_post)
        if hasattr(self, "cond_layer"): remove_weight_norm(self.cond_layer)

    @torch.no_grad()
    def inference(self, c, padding=True):
        if padding:
            c = torch.nn.functional.pad(c, (self.inference_padding, self.inference_padding), "replicate")
        return self.forward(c)
