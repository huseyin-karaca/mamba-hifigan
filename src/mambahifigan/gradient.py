"""Failure family 2: gradient and optimisation instability.

Multiplicative and deep-sequential integrations. FiLM's scale-and-shift introduces DC offsets that move zero crossings and amplifies gradient variance; squeeze-excite channel attention can only reweight features that the CNN already produced, so it cannot supply missing harmonic structure; running Mamba before the MRF lengthens the path and starves the early Mamba parameters of gradient."""

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
from .blocks import MambaModulator, MambaSEBlock, SafeMambaBlock


class HifiganGeneratorMambaFiLM(torch.nn.Module):
    """
    HiFiGAN with Mamba-Based Feature-wise Linear Modulation (FiLM).
    
    Novelty:
    Instead of adding Mamba features (residual), we use Mamba to dynamicallly
    scale and shift the convolutional features. This separates 'Structure' (Mamba)
    from 'Texture' (Conv).
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

        # 3. Mamba Modulators ve MRF Blocks
        self.mamba_modulators = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i+1))
            
            # --- NOVELTY: Mamba Modulator ---
            # Her upsample bloğu için bir yönetici Mamba
            # Hafıza yükünü azaltmak için d_state=d_state yeterli
            self.mamba_modulators.append(MambaModulator(channels=ch, d_state=d_state))
            
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

        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o) # Upsampling
            
            # --- FiLM MODULATION START ---
            # 1. Mamba'dan Scale (gamma) ve Shift (beta) al
            gamma, beta = self.mamba_modulators[i](o)
            
            # 2. Modülasyon Uygula
            # Formül: x * (1 + tanh(gamma)) + beta
            # tanh kullanıyoruz ki scale faktörü patlamasın (stabilite için)
            o_modulated = o * (1 + torch.tanh(gamma)) + beta
            # -----------------------------
            
            # 3. MRF (Texture Generation)
            # MRF blokları artık Mamba tarafından 'yönetilmiş' sinyali işler
            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i*self.num_kernels+j](o_modulated)
                else:
                    z_sum += self.resblocks[i*self.num_kernels+j](o_modulated)
            
            o = z_sum / self.num_kernels
            
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for l in self.ups: remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()
        for l in self.mamba_modulators: l.remove_weight_norm()
        remove_weight_norm(self.conv_post)
        if hasattr(self, "cond_layer"): remove_weight_norm(self.cond_layer)

    @torch.no_grad()
    def inference(self, c, padding=True):
        if padding:
            c = torch.nn.functional.pad(c, (self.inference_padding, self.inference_padding), "replicate")
        return self.forward(c)


class HifiganGeneratorMambaSE(torch.nn.Module):
    """
    HiFiGAN with Mamba-SE (Channel Attention).
    
    Design:
    - Standard HiFi-GAN Backbone (3 Kernels: 3, 7, 11) - Nothing removed!
    - After each Upsample+MRF stage, a 'MambaSEBlock' recalibrates the features.
    
    Why:
    - Preserves Texture (Conv generates the signal).
    - Adds Structure (Mamba adjusts the feature weights based on global context).
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
        self.mamba_se_blocks = nn.ModuleList() # YENİ: Attention Blokları

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
            
            # MRF (Orijinal yapıyı koruyoruz)
            for j, (k_res, d_res) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(current_ch, k_res, d_res))
            
            # --- NOVELTY: Mamba SE Block ---
            # MRF çıkışını "düzenlemek" için ekliyoruz.
            self.mamba_se_blocks.append(MambaSEBlock(channels=current_ch))

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
            
            # MRF İşlemi (Doku Üretimi)
            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i*self.num_kernels+j](o)
                else:
                    z_sum += self.resblocks[i*self.num_kernels+j](o)
            
            o = z_sum / self.num_kernels
            
            # --- MAMBA ATTENTION ---
            # "Ürettiğin özelliklerin hangileri önemli, hangileri gürültü?"
            # Mamba burada bir filtre/maske uygular.
            o = self.mamba_se_blocks[i](o)
            # -----------------------
            
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

    def remove_weight_norm(self):
        print('Removing weight norm...')
        remove_weight_norm(self.conv_pre)
        for l in self.ups: remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()
        # Mamba SE temizliği (proj layerlarında weight norm yok ama loop bozulmasın)
        for l in self.mamba_se_blocks: l.remove_weight_norm()
            
        remove_weight_norm(self.conv_post)
        if hasattr(self, "cond_layer"): remove_weight_norm(self.cond_layer)

    @torch.no_grad()
    def inference(self, c, padding=True):
        if padding:
            c = torch.nn.functional.pad(c, (self.inference_padding, self.inference_padding), "replicate")
        return self.forward(c)


class HifiganGeneratorSequentialProgressive(torch.nn.Module):
    """
    HiFi-GAN with Sequential Progressive Mamba.
    
    Logic:
    Instead of running Mamba and Conv in parallel (fighting), we run them in SERIES.
    Upsample -> Mamba (Fix Structure) -> MRF (Generate Texture).
    
    Constraints:
    - We only apply Mamba in the first 'mamba_layers' (Low Res).
    - High Res layers are pure Conv to preserve texture fidelity.
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
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        
        # STRATEJİ: Mamba kaç katmanda çalışsın?
        # Toplam katmanların yarısı kadar (Örn: 4 ise ilk 2'sinde)
        self.mamba_active_layers = self.num_upsamples // 2
        
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

        for i, (u, k) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            # A. Upsample
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
            
            # B. Mamba Block (Sadece ilk katmanlar için oluşturuyoruz)
            if i < self.mamba_active_layers:
                self.mambas.append(SafeMambaBlock(channels=ch, d_state=d_state))
            else:
                self.mambas.append(None) # Boş tutucu
            
            # C. MRF Blocks (Conv)
            for j, (k_res, d_res) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k_res, d_res))

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
            
            # 1. Upsample (Büyüt)
            o = self.ups[i](o) 
            
            # --- SEQUENTIAL MAMBA (SIRALI) ---
            # Eğer bu katmanda Mamba aktifse:
            if self.mambas[i] is not None:
                # ÖNCE Mamba çalışır ve yapıyı düzeltir
                o = self.mambas[i](o)
            # ---------------------------------
            
            # 2. MRF (Conv)
            # Mamba'nın hazırladığı/düzelttiği sinyali alıp dokusunu işler
            z_sum = None
            for j in range(self.num_kernels):
                res_out = self.resblocks[i*self.num_kernels+j](o)
                if z_sum is None:
                    z_sum = res_out
                else:
                    z_sum += res_out
            
            # MRF ortalaması o anki output olur
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
        for l in self.mambas: 
            if l is not None: l.remove_weight_norm()
            
        remove_weight_norm(self.conv_post)
        if hasattr(self, "cond_layer"): remove_weight_norm(self.cond_layer)

    @torch.no_grad()
    def inference(self, c, padding=True):
        if padding:
            c = torch.nn.functional.pad(c, (self.inference_padding, self.inference_padding), "replicate")
        return self.forward(c)
