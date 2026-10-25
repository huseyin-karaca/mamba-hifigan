"""Failure family 3: information bottlenecks.

These keep both modalities but compress or interpolate on the way. Channel concatenation followed by a 1x1 projection has to disentangle dense structure from high-frequency texture through one layer, and the optimiser favours the CNN path; broadcasting a single low-resolution context upward smears coarse features rather than creating structure; refining the mel-spectrogram before the generator cannot help, because a mel is magnitude-only and carries no phase to fix."""

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
from .blocks import BiMamba2Block, SpectrogramRefiner


class HifiganGeneratorConcat(torch.nn.Module):
    """
    HiFiGAN with Concatenation Fusion Strategy.
    
    Logic:
    Instead of adding (summing) Conv and Mamba outputs, we CONCATENATE them along the channel dimension.
    Then, a 1x1 Convolution projects them back to the original channel size.
    This preserves distinct features from both modalities without cancellation.
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
        
        # 2. Layers
        self.ups = nn.ModuleList()
        self.mambas = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        self.fusion_projs = nn.ModuleList() # Kanal sayısını düşürmek için 1x1 Convlar

        for i, (u, k) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            # A. Upsample Layer
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
            
            # B. Mamba Block
            self.mambas.append(BiMamba2Block(channels=current_ch, d_state=d_state))
            
            # C. ResBlocks (MRF)
            for j, (k_res, d_res) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(current_ch, k_res, d_res))
            
            # D. Fusion Projection (1x1 Conv)
            # Girdi Kanalları = (Conv Sayısı * ch) + (Mamba * ch)
            # ResBlock sayısı genelde 3'tür (num_kernels). Mamba 1 tane.
            # Toplam giriş kanalı = (num_kernels + 1) * current_ch
            total_concat_channels = (self.num_kernels + 1) * current_ch
            
            self.fusion_projs.append(weight_norm(
                nn.Conv1d(
                    in_channels=total_concat_channels,
                    out_channels=current_ch,
                    kernel_size=1 # 1x1 Conv sadece kanalları karıştırır
                )
            ))

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
            o = self.ups[i](o) # Upsampling
            
            # --- CONCATENATION STRATEGY ---
            features_to_concat = []
            
            # 1. Conv (MRF) Çıktılarını Topla
            for j in range(self.num_kernels):
                # ResBlock'lar o'yu alıp işler
                conv_feat = self.resblocks[i*self.num_kernels+j](o)
                features_to_concat.append(conv_feat)
            
            # 2. Mamba Çıktısını Topla
            mamba_feat = self.mambas[i](o)
            features_to_concat.append(mamba_feat)
            
            # 3. Hepsini Yapıştır (Concatenate)
            # Boyut: [Batch, (Kernels+1)*Ch, Time]
            concat_out = torch.cat(features_to_concat, dim=1)
            
            # 4. Projeksiyon ile Boyutu Geri Düşür
            # Boyut: [Batch, Ch, Time]
            o = self.fusion_projs[i](concat_out)
            # ------------------------------
            
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
        for l in self.fusion_projs: remove_weight_norm(l)
            
        remove_weight_norm(self.conv_post)
        if hasattr(self, "cond_layer"): remove_weight_norm(self.cond_layer)

    @torch.no_grad()
    def inference(self, c, padding=True):
        if padding:
            c = torch.nn.functional.pad(c, (self.inference_padding, self.inference_padding), "replicate")
        return self.forward(c)


class HifiganGeneratorBroadcast(torch.nn.Module):
    """
    HiFiGAN with Global Context Broadcasting.
    
    Novelty:
    - Main Stream: Pure HiFi-GAN (Preserves Texture/FM Loss).
    - Side Stream: Mamba processes the Mel-Spec ONCE at the start.
    - Broadcasting: Mamba features are upsampled and injected into EVERY layer of HiFi-GAN.
    
    This avoids 'oversmoothing' because the waveform never passes THROUGH Mamba layers.
    Mamba acts as a read-only 'Conductor' giving cues to the Conv 'Musicians'.
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
        
        # --- KANAL KİLİDİ ---
        in_channels = 80 
        # --------------------

        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2

        # --- 1. SIDE STREAM (Mamba Context Encoder) ---
        # Girişteki Mel-Spectrogram'ı işleyip güçlü bir context çıkarır.
        # d_model=128 veya 256 olabilir, çünkü sadece 1 kez çalışacak.
        self.mamba_context_encoder = nn.Sequential(
             weight_norm(nn.Conv1d(in_channels, 128, 7, 1, padding=3)),
             BiMamba2Block(channels=128, d_state=d_state),
             BiMamba2Block(channels=128, d_state=d_state)
        )
        # ----------------------------------------------

        # 2. Main Stream Initial Conv
        self.conv_pre = weight_norm(nn.Conv1d(
            in_channels=in_channels,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3
        ))
        
        # 3. Layers
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        self.context_projections = nn.ModuleList() # Mamba bilgisini Conv boyutuna uydurmak için

        for i, (u, k) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            # Upsample Layer
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
            
            # MRF Blocks (Orijinal HiFi-GAN, hiçbir şeyi silmiyoruz!)
            for j, (k_res, d_res) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(current_ch, k_res, d_res))
            
            # --- BROADCAST ADAPTER ---
            # Mamba'nın çıkışı (128) -> O anki Conv kanalı (current_ch)
            self.context_projections.append(weight_norm(
                nn.Conv1d(128, current_ch, 1)
            ))

        # 4. Post Conv
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
        # x: [Batch, 80, Time_Mel]
        
        # --- ADIM 1: GLOBAL CONTEXT OLUŞTUR (Yan Hat) ---
        # Mel-spectrogram'ı Mamba ile işle.
        # Bu context, tüm cümlenin yapısını bilir.
        global_context = self.mamba_context_encoder(x) 
        # global_context: [Batch, 128, Time_Mel]
        # ------------------------------------------------

        # --- ADIM 2: ANA HAT (Main Stream) ---
        o = self.conv_pre(x)
        if hasattr(self, "cond_layer"):
            o = o + self.cond_layer(g)

        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            
            # Upsample (Boyut Büyür)
            o = self.ups[i](o)
            # o boyutu: [Batch, Ch, Time_HighRes]
            
            # --- BROADCASTING (ENJEKSİYON) ---
            # Mamba Context'ini o anki boyuta uydur (Interpolate)
            ctx_upsampled = F.interpolate(
                global_context, 
                size=o.shape[2], # Hedef zaman boyutu
                mode='linear',   # Veya 'nearest'
                align_corners=False
            )
            
            # Kanal sayısını uydur (Projection)
            ctx_projected = self.context_projections[i](ctx_upsampled)
            
            # Ana hatta ekle (Injection)
            o = o + ctx_projected
            # ---------------------------------
            
            # MRF İşlemleri (Texture Generation)
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
        
        # Mamba Encoder Temizliği
        for layer in self.mamba_context_encoder:
            if hasattr(layer, "remove_weight_norm"): layer.remove_weight_norm()
            if isinstance(layer, BiMamba2Block): layer.remove_weight_norm()
            
        for l in self.context_projections: remove_weight_norm(l)
            
        remove_weight_norm(self.conv_post)
        if hasattr(self, "cond_layer"): remove_weight_norm(self.cond_layer)

    @torch.no_grad()
    def inference(self, c, padding=True):
        if padding:
            c = torch.nn.functional.pad(c, (self.inference_padding, self.inference_padding), "replicate")
        return self.forward(c)


class HifiganGeneratorWithPreNet(torch.nn.Module):
    """
    HiFiGAN with Mamba Context Pre-Net.
    
    Strategy:
    Instead of interfering with the upsampling layers (which hurts texture/FM Loss),
    we use Mamba ONLY at the beginning (low resolution) to inject global context.
    The rest is standard HiFi-GAN to preserve high-fidelity texture.
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
        
        # --- ACİL DURUM: Kanal Sayısı Kilidi ---
        in_channels = 80
        # ---------------------------------------

        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        
        # ResBlock Import
        from speechbrain.lobes.models.HifiGAN import ResBlock1, ResBlock2
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2

        # 1. Initial Convolution
        self.conv_pre = weight_norm(nn.Conv1d(
            in_channels=in_channels,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3
        ))
        
        # --- 2. MAMBA PRE-NET (YENİLİK BURADA) ---
        # Sinyali bozmadan sadece "zenginleştirmek" için Residual Mamba kullanıyoruz.
        # Düşük çözünürlükte (Mel seviyesinde) çalıştığı için çok güçlüdür.
        self.mamba_prenet = BiMamba2Block(
            channels=upsample_initial_channel, 
            d_state=d_state  # Hafıza kapasitesi
        )
        # -----------------------------------------

        # 3. Standard HiFi-GAN Upsampling (Orijinal Yapı)
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

        # 4. Standard HiFi-GAN MRF Blocks (Doku için)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        # 5. Post Convolution
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
        # 1. Giriş
        o = self.conv_pre(x)
        
        if hasattr(self, "cond_layer"):
            o = o + self.cond_layer(g)

        # --- MAMBA CONTEXT INJECTION ---
        # Sadece burada Mamba çalışır. Tüm cümlenin bağlamını yakalar.
        # Residual bağlantı (o + mamba) sayesinde orijinal sinyali bozmaz, sadece düzeltir.
        mamba_out = self.mambas_prenet(o) if hasattr(self, 'mambas_prenet') else self.mamba_prenet(o)
        o = o + mamba_out
        # -------------------------------

        # 2. HiFi-GAN Gövdesi (Buraya dokunmuyoruz, texture bozulmasın diye)
        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o) # Upsample
            
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
        # Mamba Pre-Net temizliği
        if hasattr(self.mamba_prenet, "remove_weight_norm"):
            self.mamba_prenet.remove_weight_norm()
            
        remove_weight_norm(self.conv_pre)
        for l in self.ups: remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()
        remove_weight_norm(self.conv_post)
        if hasattr(self, "cond_layer"): remove_weight_norm(self.cond_layer)

    @torch.no_grad()
    def inference(self, c, padding=True):
        if padding:
            c = torch.nn.functional.pad(c, (self.inference_padding, self.inference_padding), "replicate")
        return self.forward(c)


class HifiganGeneratorSpecRefine(torch.nn.Module):
    """
    HiFi-GAN with Input Refinement.
    
    Novelty:
    Instead of putting Mamba INSIDE the generator (where it fights with Conv),
    we put Mamba BEFORE the generator. 
    It 'fixes' the Mel-Spectrogram using global context.
    
    Architecture:
    Input -> [Mamba Refiner] -> Refined_Input -> [Standard HiFi-GAN] -> Audio
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
        
        # --- PART 1: MAMBA REFINER (YENİLİK BURADA) ---
        # Girişteki 80 kanalı işler. HiFi-GAN'a girmeden önce veriyi zenginleştirir.
        self.input_refiner = SpectrogramRefiner(channels=in_channels, d_state=d_state, layers=2)
        # ----------------------------------------------

        # --- PART 2: STANDARD HiFi-GAN (DOKUNULMAMIŞ) ---
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2

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

        self.apply(init_weights)

    def forward(self, x, g=None):
        # x: [Batch, 80, Time]
        
        # --- ADIM 1: REFINEMENT ---
        # Spektrogramı Mamba ile düzelt
        x = self.input_refiner(x)
        # --------------------------
        
        # --- ADIM 2: STANDARD GENERATION ---
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
        o = torch.tanh(o)
        return o

    def remove_weight_norm(self):
        print('Removing weight norm...')
        remove_weight_norm(self.conv_pre)
        for l in self.ups: remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()
        remove_weight_norm(self.conv_post)
        if hasattr(self, "cond_layer"): remove_weight_norm(self.cond_layer)
        
        # Refiner temizliği
        self.input_refiner.remove_weight_norm()

    @torch.no_grad()
    def inference(self, c, padding=True):
        if padding:
            c = torch.nn.functional.pad(c, (self.inference_padding, self.inference_padding), "replicate")
        return self.forward(c)
