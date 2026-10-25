"""The Mamba building blocks.

`BiMamba2Block` is the one that matters: it is used by every configuration that reached the results table. The
others exist because a particular integration strategy needed them, and each is named for that strategy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.lobes.models.HifiGAN import LRELU_SLOPE

try:
    from mamba_ssm import Mamba2
except ImportError:  # the blocks that need it raise on construction
    Mamba2 = None


class BiMamba2Block(nn.Module):
    """Bidirectional Mamba2 Block"""

    def __init__(self, channels, d_state=64, d_conv=4, expand=2):
        super().__init__()
        if Mamba2 is None:
            raise ImportError("mamba_ssm is not installed.")
        self.forward_mamba = Mamba2(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.backward_mamba = Mamba2(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: [Batch, Channel, Time]
        x_t = x.transpose(1, 2)  # [Batch, Time, Channel]

        out_fwd = self.forward_mamba(x_t)
        out_bwd = self.backward_mamba(x_t.flip(1)).flip(1)

        out = out_fwd + out_bwd
        out = self.norm(out)

        return out.transpose(1, 2)  # [Batch, Channel, Time]

    def remove_weight_norm(self):
        # HiFi-GAN inference sırasında hata vermemesi için boş fonksiyon
        pass


class SafeMambaBlock(nn.Module):
    """
    Memory-Safe Mamba Block.
    Projects input to 64 dims to fit in Shared Memory and avoid errors.
    """
    def __init__(self, channels, d_state=64):
        super().__init__()
        if Mamba2 is None: raise ImportError("mamba_ssm required")
        
        self.mamba_dim = 64 # Güvenli boyut
        
        # Projection: Input Channel -> 64
        self.input_proj = weight_norm(nn.Conv1d(channels, self.mamba_dim, 1))
        
        self.mamba = Mamba2(
            d_model=self.mamba_dim, 
            d_state=d_state, 
            d_conv=4, 
            expand=1, 
            headdim=64 
        )
        self.norm = nn.LayerNorm(self.mamba_dim)
        
        # Projection: 64 -> Input Channel
        self.output_proj = weight_norm(nn.Conv1d(self.mamba_dim, channels, 1))

    def forward(self, x):
        # x: [B, C, T]
        residual = x
        x = self.input_proj(x)
        x_t = x.transpose(1, 2)
        x_out = self.mamba(x_t)
        x_out = self.norm(x_out).transpose(1, 2)
        x_out = self.output_proj(x_out)
        return residual + x_out # Residual connection
        
    def remove_weight_norm(self): 
        remove_weight_norm(self.input_proj)
        remove_weight_norm(self.output_proj)


class LowResMambaBlock(nn.Module):
    """
    Robust Multi-Rate Mamba Block.
    
    FIX: 
    - Input channels are projected to a fixed 'mamba_dim' (64).
    - This guarantees 'headdim=64' which fits in ANY GPU Shared Memory.
    - Solves both 'AssertionError' and 'Out of Resources' errors.
    """
    def __init__(self, channels, d_state=64, scale_factor=2):
        super().__init__()
        if Mamba2 is None: raise ImportError("mamba_ssm required")
        
        self.scale_factor = scale_factor
        
        # 1. Downsample (Pooling)
        self.pool = nn.AvgPool1d(kernel_size=scale_factor, stride=scale_factor)
        
        # --- FIXED PROJECTION ---
        # Mamba'nın içinde çalışacağı güvenli boyut (64 her GPU'ya sığar)
        self.mamba_dim = 64 
        
        # Giriş Kanalı -> Güvenli Mamba Boyutu (64)
        self.input_proj = weight_norm(nn.Conv1d(channels, self.mamba_dim, 1))
        
        # 2. Mamba Core
        # Artık headdim=64 sabit, d_model=64 sabit. Hata verme şansı yok.
        self.mamba = Mamba2(
            d_model=self.mamba_dim, 
            d_state=d_state, 
            d_conv=4, 
            expand=1, 
            headdim=64 
        )
        self.norm = nn.LayerNorm(self.mamba_dim)
        
        # 3. Output Projection (Geri Dönüş)
        # Güvenli Mamba Boyutu (64) -> Orijinal Kanal Sayısı
        self.output_proj = weight_norm(nn.Conv1d(self.mamba_dim, channels, 1))

    def forward(self, x):
        # x: [Batch, Channel, Time]
        residual = x
        
        # A. Downsample (Low Freq)
        x_low = self.pool(x)
        
        # B. Project to Safe Dim (Channel -> 64)
        x_low = self.input_proj(x_low)
        
        # C. Mamba Process
        # [B, 64, T_low] -> [B, T_low, 64]
        x_low_t = x_low.transpose(1, 2)
        x_out = self.mamba(x_low_t)
        x_out = self.norm(x_out).transpose(1, 2)
        
        # D. Project Back (64 -> Channel)
        x_out = self.output_proj(x_out)
        
        # E. Upsample
        x_up = F.interpolate(x_out, size=x.shape[2], mode='linear', align_corners=False)
        
        # F. Residual Add
        return residual + x_up
        
    def remove_weight_norm(self): 
        remove_weight_norm(self.input_proj)
        remove_weight_norm(self.output_proj)


class MambaRefiner(nn.Module):
    """
    A standalone Mamba module that takes Raw Audio and refines it.
    Input: [Batch, 1, Time]
    Output: [Batch, 1, Time] (Residual Correction)
    """
    def __init__(self, d_model=64):
        super().__init__()
        if Mamba2 is None: raise ImportError("mamba_ssm required")
        
        # 1. Projection: Audio (1 kanal) -> Feature (d_model)
        self.input_proj = weight_norm(nn.Conv1d(1, d_model, kernel_size=7, padding=3))
        
        # 2. Deep Mamba Layers (Global Context Cleaning)
        # Sinyal çok uzun olduğu için d_state küçük tutulmalı (16 veya 32)
        self.mamba1 = Mamba2(d_model=d_model, d_state=32, d_conv=4, expand=2)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.mamba2 = Mamba2(d_model=d_model, d_state=32, d_conv=4, expand=2)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 3. Output Projection: Feature -> Audio Correction (1 kanal)
        self.output_proj = weight_norm(nn.Conv1d(d_model, 1, kernel_size=7, padding=3))
        
        # Zero Init (Model eğitime "etkisiz" başlasın diye)
        self.output_proj.weight.data.fill_(0.0)
        self.output_proj.bias.data.fill_(0.0)

    def forward(self, x):
        # x: [Batch, 1, Time]
        residual = x
        
        # Feature Extraction
        x_feat = self.input_proj(x) # [B, C, T]
        
        # Mamba Processing
        x_t = x_feat.transpose(1, 2) # [B, T, C]
        
        x_t = self.mamba1(x_t)
        x_t = self.norm1(x_t)
        x_t = F.leaky_relu(x_t, LRELU_SLOPE)
        
        x_t = self.mamba2(x_t)
        x_t = self.norm2(x_t)
        x_t = F.leaky_relu(x_t, LRELU_SLOPE)
        
        x_back = x_t.transpose(1, 2) # [B, C, T]
        
        # Correction Signal
        correction = self.output_proj(x_back)
        
        # Add Correction to Original Audio
        return residual + correction

    def remove_weight_norm(self):
        remove_weight_norm(self.input_proj)
        remove_weight_norm(self.output_proj)


class MambaModulator(nn.Module):
    """
    Mamba based FiLM Generator.
    Input: [B, C, T]
    Output: Gamma (Scale) [B, C, T] and Beta (Shift) [B, C, T]
    """
    def __init__(self, channels, d_state=64):
        super().__init__()
        if Mamba2 is None:
            raise ImportError("mamba_ssm is required")
        
        # Mamba Core
        self.forward_mamba = Mamba2(d_model=channels, d_state=d_state, d_conv=4, expand=2)
        self.backward_mamba = Mamba2(d_model=channels, d_state=d_state, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(channels)
        
        # Projection to 2x Channels (Gamma + Beta)
        # 1x1 Conv ile kanal sayısını ikiye katlıyoruz.
        self.proj = nn.Conv1d(channels, channels * 2, kernel_size=1)

    def forward(self, x):
        # x: [B, C, T]
        x_t = x.transpose(1, 2)
        out = self.forward_mamba(x_t) + self.backward_mamba(x_t.flip(1)).flip(1)
        out = self.norm(out).transpose(1, 2) # [B, C, T]
        
        # Scale ve Shift üret
        stats = self.proj(out) # [B, 2*C, T]
        gamma, beta = stats.chunk(2, dim=1)
        
        return gamma, beta
    
    def remove_weight_norm(self):
        pass


class MambaSEBlock(nn.Module):
    """
    Mamba-based Squeeze-and-Excitation (SE) Block.
    FIXED: Added 'headdim' parameter to prevent Mamba2 assertion errors on small channels.
    """
    def __init__(self, channels, d_state=16):
        super().__init__()
        if Mamba2 is None: raise ImportError("mamba_ssm required")
        
        # Kanal sayısını azaltıp (Squeeze) işlem maliyetini düşürüyoruz
        self.reduction_ratio = 4
        # En az 8 kanala düşsün, daha aşağısı olmasın
        reduced_channels = max(channels // self.reduction_ratio, 8)
        
        self.input_proj = nn.Conv1d(channels, reduced_channels, 1)
        
        # --- FIX BURADA ---
        # Mamba2 varsayılan headdim=64'tür. 
        # Reduced channels 64'ten küçükse (örn: 32, 16, 8) hata verir.
        # Bu yüzden headdim'i reduced_channels'a eşitliyoruz.
        self.mamba = Mamba2(
            d_model=reduced_channels, 
            d_state=d_state, 
            d_conv=2, 
            expand=1,
            headdim=reduced_channels  # <--- BU SATIR HATAYI ÇÖZER
        )
        # ------------------
        
        self.output_proj = nn.Conv1d(reduced_channels, channels, 1)
        self.act = nn.Sigmoid() 

    def forward(self, x):
        # x: [Batch, Channel, Time]
        residual = x
        
        # 1. Squeeze
        x_squeezed = self.input_proj(x)
        
        # 2. Mamba Process
        # [B, C, T] -> [B, T, C]
        x_mamba = self.mamba(x_squeezed.transpose(1, 2)).transpose(1, 2)
        
        # 3. Excitation
        weights = self.output_proj(x_mamba)
        weights = self.act(weights) 
        
        # 4. Re-weight
        return residual * weights
        
    def remove_weight_norm(self): 
        pass


class MambaSideNetwork(nn.Module):
    """
    A lightweight, Mamba-based generator that runs in parallel.
    It produces a 'Correction Waveform'.
    """
    def __init__(self, in_channels, upsample_factors, upsample_initial_channel):
        super().__init__()
        
        # Giriş Katmanı
        self.conv_pre = weight_norm(nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3))
        
        self.ups = nn.ModuleList()
        self.mambas = nn.ModuleList()
        
        # Basitleştirilmiş Upsample + Mamba yapısı (Conv MRF yok, sadece Mamba)
        for i, u in enumerate(upsample_factors):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2**(i+1)),
                    kernel_size=u*2, # Basit kernel
                    stride=u,
                    padding=u // 2
                )
            ))
            
            ch = upsample_initial_channel // (2**(i+1))
            # Her adımda yapıyı korumak için Mamba
            self.mambas.append(SafeMambaBlock(channels=ch, d_state=32)) # Hafif Mamba
            
        # Çıkış Katmanı (Waveform üretir)
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        
        # --- ZERO INITIALIZATION (SİHİR BURADA) ---
        # Bu katmanın ağırlıklarını 0 yapıyoruz.
        # Böylece eğitimin başında bu ağ "0" (sessizlik) üretir.
        # Model, Baseline HiFi-GAN ile aynı performansta başlar.
        self.conv_post.weight.data.fill_(0.0)
        self.conv_post.bias.data.fill_(0.0)

    def forward(self, x):
        o = self.conv_pre(x)
        for i in range(len(self.ups)):
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o)
            o = self.mambas[i](o)
        
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        return torch.tanh(o)
        
    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for l in self.ups: remove_weight_norm(l)
        for l in self.mambas: l.remove_weight_norm()
        remove_weight_norm(self.conv_post)


class SpectrogramRefiner(nn.Module):
    """
    FIXED Spectrogram Refiner.
    
    Problem Solved: 
    - Mel-Spec input is 80 channels.
    - Mamba with expand=2 creates internal dim 160.
    - 160 is not divisible by headdim=64 (AssertionError).
    
    Solution:
    - We project Input(80) -> SafeDim(64) before Mamba.
    - Mamba runs smoothly on 64 dims.
    - We project SafeDim(64) -> Output(80) to get the residual.
    """
    def __init__(self, channels=80, d_state=64, layers=2):
        super().__init__()
        if Mamba2 is None: raise ImportError("mamba_ssm required")
        
        # Mamba için Güvenli Boyut (64 her şeye bölünür ve GPU dostudur)
        self.safe_dim = 64
        
        # 1. Input Projection (80 -> 64)
        self.in_proj = weight_norm(nn.Conv1d(channels, self.safe_dim, 1))
        
        # 2. Mamba Layers (Running on 64 channels)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(
                Mamba2(
                    d_model=self.safe_dim, # Artık 64, sorunsuz.
                    d_state=d_state, 
                    d_conv=4, 
                    expand=2, 
                    headdim=64 # 64 * 2 = 128, 128 % 64 == 0. (Hata Yok)
                )
            )
            self.layers.append(nn.LayerNorm(self.safe_dim))

        # 3. Output Projection (64 -> 80) - Zero Init
        self.out_proj = weight_norm(nn.Conv1d(self.safe_dim, channels, 1))
        
        # Zero-Init: Başlangıçta girdiye '0' ekler, baseline ile aynı başlar.
        self.out_proj.weight.data.fill_(0.0)
        self.out_proj.bias.data.fill_(0.0)

    def forward(self, x):
        # x: [Batch, 80, T]
        residual_in = x
        
        # Project to Safe Dim
        x_safe = self.in_proj(x) # [B, 64, T]
        
        # Mamba Process
        x_t = x_safe.transpose(1, 2) # [B, T, 64]
        
        for layer in self.layers:
            if isinstance(layer, Mamba2):
                # Bidirectional trick
                x_fwd = layer(x_t)
                x_bwd = layer(x_t.flip(1)).flip(1)
                x_t = x_fwd + x_bwd
            else:
                x_t = layer(x_t) # Norm
        
        x_back = x_t.transpose(1, 2) # [B, 64, T]
        
        # Project Back to Original Dim (Calculate Correction)
        delta = self.out_proj(x_back) # [B, 80, T]
        
        # Add Correction to Original Input
        return residual_in + delta
    
    def remove_weight_norm(self):
        remove_weight_norm(self.in_proj)
        remove_weight_norm(self.out_proj)
