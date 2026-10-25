"""
Neural network modules for the HiFi-GAN: Generative Adversarial Networks for
Efficient and High Fidelity Speech Synthesis

For more details: https://arxiv.org/pdf/2010.05646.pdf, https://arxiv.org/abs/2406.10735

Authors
 * Jarod Duret 2021
 * Yingzhi WANG 2022
"""

# Adapted from https://github.com/jik876/hifi-gan/ and https://github.com/coqui-ai/TTS/
# MIT License

# Copyright (c) 2020 Jungil Kong

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms

import speechbrain as sb
from speechbrain.nnet.CNN import Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

# --- 1. YARDIMCI FONKSİYONLAR (Dependency Yok) ---
def init_weights(m, mean=0.0, std=0.01):
    """
    Saf PyTorch katmanları için ağırlık başlatma.
    SpeechBrain wrapper'ı olmadığı için 'weight' attribute'u kesin vardır.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.normal_(mean, std)
        # Weight norm uygulanmışsa (g ve v parametreleri)
        if hasattr(m, "weight_g") and m.weight_g is not None:
            m.weight_g.data.normal_(mean, std)
        if hasattr(m, "weight_v") and m.weight_v is not None:
            m.weight_v.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None
    print("Warning: mamba_ssm not found. Mamba2 will not be available.")

LRELU_SLOPE = 0.1


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """Dynamique range compression for audio signals"""
    return torch.log(torch.clamp(x, min=clip_val) * C)


def mel_spectogram(
    sample_rate,
    hop_length,
    win_length,
    n_fft,
    n_mels,
    f_min,
    f_max,
    power,
    normalized,
    norm,
    mel_scale,
    compression,
    audio,
):
    """calculates MelSpectrogram for a raw audio signal

    Arguments
    ---------
    sample_rate : int
        Sample rate of audio signal.
    hop_length : int
        Length of hop between STFT windows.
    win_length : int
        Window size.
    n_fft : int
        Size of FFT.
    n_mels : int
        Number of mel filterbanks.
    f_min : float
        Minimum frequency.
    f_max : float
        Maximum frequency.
    power : float
        Exponent for the magnitude spectrogram.
    normalized : bool
        Whether to normalize by magnitude after stft.
    norm : str or None
        If "slaney", divide the triangular mel weights by the width of the mel band
    mel_scale : str
        Scale to use: "htk" or "slaney".
    compression : bool
        whether to do dynamic range compression
    audio : torch.tensor
        input audio signal

    Returns
    -------
    Mel spectrogram
    """

    audio_to_mel = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=power,
        normalized=normalized,
        norm=norm,
        mel_scale=mel_scale,
    ).to(audio.device)

    mel = audio_to_mel(audio)

    if compression:
        mel = dynamic_range_compression(mel)

    return mel


def process_duration(code, code_feat):
    """
    Process a given batch of code to extract consecutive unique elements and their associated features.

    Arguments
    ---------
    code : torch.Tensor (batch, time)
        Tensor of code indices.
    code_feat : torch.Tensor (batch, time, channel)
        Tensor of code features.

    Returns
    -------
    uniq_code_feat_filtered : torch.Tensor (batch, time)
        Features of consecutive unique codes.
    mask : torch.Tensor (batch, time)
        Padding mask for the unique codes.
    uniq_code_count : torch.Tensor (n)
        Count of unique codes.

    Example
    -------
    >>> code = torch.IntTensor([[40, 18, 18, 10]])
    >>> code_feat = torch.rand([1, 4, 128])
    >>> out_tensor, mask, uniq_code = process_duration(code, code_feat)
    >>> out_tensor.shape
    torch.Size([1, 1, 128])
    >>> mask.shape
    torch.Size([1, 1])
    >>> uniq_code.shape
    torch.Size([1])
    """
    uniq_code_count = []
    uniq_code_feat = []
    for i in range(code.size(0)):
        _, count = torch.unique_consecutive(code[i, :], return_counts=True)
        if len(count) > 2:
            # remove first and last code as segment sampling may cause incomplete segment length
            uniq_code_count.append(count[1:-1])
            uniq_code_idx = count.cumsum(dim=0)[:-2]
        else:
            uniq_code_count.append(count)
            uniq_code_idx = count.cumsum(dim=0) - 1
        uniq_code_feat.append(
            code_feat[i, uniq_code_idx, :].view(-1, code_feat.size(2))
        )
    uniq_code_count = torch.cat(uniq_code_count)

    # collate
    max_len = max(feat.size(0) for feat in uniq_code_feat)
    uniq_code_feat_filtered = uniq_code_feat[0].new_zeros(
        (len(uniq_code_feat), max_len, uniq_code_feat[0].size(1))
    )
    mask = torch.arange(max_len).repeat(len(uniq_code_feat), 1)
    for i, v in enumerate(uniq_code_feat):
        uniq_code_feat_filtered[i, : v.size(0)] = v
        mask[i, :] = mask[i, :] < v.size(0)

    return uniq_code_feat_filtered, mask.bool(), uniq_code_count.float()


##################################
# Generator
##################################


class ResBlock1(torch.nn.Module):
    """
    Residual Block Type 1, which has 3 convolutional layers in each convolution block.

    Arguments
    ---------
    channels : int
        number of hidden channels for the convolutional layers.
    kernel_size : int
        size of the convolution filter in each layer.
    dilation : list
        list of dilation value for each conv layer in a block.
    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation[0],
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation[1],
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation[2],
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=1,
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=1,
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=1,
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
            ]
        )

    def forward(self, x):
        """Returns the output of ResBlock1

        Arguments
        ---------
        x : torch.Tensor (batch, channel, time)
            input tensor.

        Returns
        -------
        The ResBlock outputs
        """

        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        """This functions removes weight normalization during inference."""
        for layer in self.convs1:
            layer.remove_weight_norm()
        for layer in self.convs2:
            layer.remove_weight_norm()


class ResBlock2(torch.nn.Module):
    """
    Residual Block Type 2, which has 2 convolutional layers in each convolution block.

    Arguments
    ---------
    channels : int
        number of hidden channels for the convolutional layers.
    kernel_size : int
        size of the convolution filter in each layer.
    dilation : list
        list of dilation value for each conv layer in a block.
    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation[0],
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation[1],
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
            ]
        )

    def forward(self, x):
        """Returns the output of ResBlock1

        Arguments
        ---------
        x : torch.Tensor (batch, channel, time)
            input tensor.

        Returns
        -------
        The ResBlock outputs
        """

        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        """This functions removes weight normalization during inference."""
        for layer in self.convs:
            layer.remove_weight_norm()


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
        self.input_refiner = SpectrogramRefiner(channels=in_channels, d_state=64, layers=2)
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

# --- HEDEF CLASS: DUAL-PATH GENERATOR ---

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





# --- HEDEF CLASS: SEQUENTIAL PROGRESSIVE ---

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
                self.mambas.append(SafeMambaBlock(channels=ch, d_state=64))
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

# --- HEDEF CLASS: MULTI-RATE HYBRID ---

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

# --- HEDEF CLASS: POST-NET ARCHITECTURE ---

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
             BiMamba2Block(channels=128, d_state=64),
             BiMamba2Block(channels=128, d_state=64)
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
            self.mambas.append(BiMamba2Block(channels=ch, d_state=64))
            
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
            self.mambas.append(BiMamba2Block(channels=current_ch, d_state=128))
            
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
            # Hafıza yükünü azaltmak için d_state=64 yeterli
            self.mamba_modulators.append(MambaModulator(channels=ch, d_state=128))
            
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
                self.mambas.append(BiMamba2Block(channels=ch, d_state=64))
            
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
                self.mambas.append(BiMamba2Block(channels=ch, d_state=128))
            
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
            d_state=128  # Hafıza kapasitesi
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
            self.mambas.append(BiMamba2Block(channels=ch, d_state=128))
            
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









class HifiganGenerator(torch.nn.Module):
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
            self.mambas.append(BiMamba2Block(channels=ch, d_state=128))
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
            self.mambas.append(BiMamba2Block(channels=ch, d_state=128))
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



class OriginalHifiganGenerator(torch.nn.Module):
    """HiFiGAN Generator with Multi-Receptive Field Fusion (MRF)

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
            o = z_sum / self.num_kernels
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
    mamba_d_state : int
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
        mamba_d_state=64,
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
            self.mambas.append(BiMamba2Block(channels=out_ch, d_state=mamba_d_state))

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
    mamba_d_state : int
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
        mamba_d_state=64,
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
            self.mambas.append(BiMamba2Block(channels=ch, d_state=mamba_d_state))
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


class VariancePredictor(nn.Module):
    """Variance predictor inspired from FastSpeech2

    Arguments
    ---------
    encoder_embed_dim : int
        number of input tensor channels.
    var_pred_hidden_dim : int
        size of hidden channels for the convolutional layers.
    var_pred_kernel_size : int
        size of the convolution filter in each layer.
    var_pred_dropout : float
        dropout probability of each layer.

    Example
    -------
    >>> inp_tensor = torch.rand([4, 80, 128])
    >>> duration_predictor = VariancePredictor(
    ...     encoder_embed_dim=128,
    ...     var_pred_hidden_dim=128,
    ...     var_pred_kernel_size=3,
    ...     var_pred_dropout=0.5,
    ... )
    >>> out_tensor = duration_predictor(inp_tensor)
    >>> out_tensor.shape
    torch.Size([4, 80])
    """

    def __init__(
        self,
        encoder_embed_dim,
        var_pred_hidden_dim,
        var_pred_kernel_size,
        var_pred_dropout,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            Conv1d(
                in_channels=encoder_embed_dim,
                out_channels=var_pred_hidden_dim,
                kernel_size=var_pred_kernel_size,
                padding="same",
                skip_transpose=True,
                weight_norm=True,
            ),
            nn.ReLU(),
        )
        self.dropout = var_pred_dropout
        self.conv2 = nn.Sequential(
            Conv1d(
                in_channels=var_pred_hidden_dim,
                out_channels=var_pred_hidden_dim,
                kernel_size=var_pred_kernel_size,
                padding="same",
                skip_transpose=True,
                weight_norm=True,
            ),
            nn.ReLU(),
        )
        self.proj = nn.Linear(var_pred_hidden_dim, 1)

    def forward(self, x):
        """
        Arguments
        ---------
        x : torch.Tensor (batch, channel, time)
            feature input tensor.

        Returns
        -------
        Variance predictor output
        """
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.proj(x).squeeze(dim=2)


class UnitHifiganGenerator(HifiganGenerator):
    """The UnitHiFiGAN generator takes discrete speech tokens as input.
    The generator is adapted to support bitrate scalability training.
    For more details, refer to: https://arxiv.org/abs/2406.10735.

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
        Whether to add a conv to the front
    conv_post_bias : bool
        Whether to add a bias to the last conv
    vocab_size : int
        size of the dictionary of embeddings.
    embedding_dim : int
        size of each embedding vector.
    attn_dim : int
        size of attention dimension.
    duration_predictor : bool
        enable duration predictor module.
    var_pred_hidden_dim : int
        size of hidden channels for the convolutional layers of the duration predictor.
    var_pred_kernel_size : int
        size of the convolution filter in each layer of the duration predictor.
    var_pred_dropout : float
        dropout probability of each layer in the duration predictor.
    multi_speaker : bool
        enable multi speaker training.
    normalize_speaker_embeddings: bool
        enable normalization of speaker embeddings.
    skip_token_embedding: bool
        Whether to skip the embedding layer in the case of continuous input.
    pooling_type: str, optional
        The type of pooling to use. Must be one of ["attention", "sum", "none"].
        Defaults to "attention" for scalable vocoder.

    Example
    -------
    >>> inp_tensor = torch.randint(0, 100, (4, 10, 1))
    >>> unit_hifigan_generator = UnitHifiganGenerator(
    ...     in_channels=128,
    ...     out_channels=1,
    ...     resblock_type="1",
    ...     resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ...     resblock_kernel_sizes=[3, 7, 11],
    ...     upsample_kernel_sizes=[11, 8, 8, 4, 4],
    ...     upsample_initial_channel=512,
    ...     upsample_factors=[5, 4, 4, 2, 2],
    ...     vocab_size=100,
    ...     embedding_dim=128,
    ...     duration_predictor=True,
    ...     var_pred_hidden_dim=128,
    ...     var_pred_kernel_size=3,
    ...     var_pred_dropout=0.5,
    ... )
    >>> out_tensor, _ = unit_hifigan_generator(inp_tensor)
    >>> out_tensor.shape
    torch.Size([4, 1, 3200])
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
        vocab_size=100,
        embedding_dim=128,
        attn_dim=128,
        duration_predictor=False,
        var_pred_hidden_dim=128,
        var_pred_kernel_size=3,
        var_pred_dropout=0.5,
        multi_speaker=False,
        normalize_speaker_embeddings=False,
        skip_token_embedding=False,
        pooling_type="attention",
    ):
        super().__init__(
            in_channels,
            out_channels,
            resblock_type,
            resblock_dilation_sizes,
            resblock_kernel_sizes,
            upsample_kernel_sizes,
            upsample_initial_channel,
            upsample_factors,
            inference_padding,
            cond_channels,
            conv_post_bias,
        )
        self.unit_embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.pooling_type = pooling_type
        if pooling_type == "attention":
            self.attn_pooling = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, attn_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(attn_dim, 1, bias=False),
            )

        self.duration_predictor = duration_predictor
        if duration_predictor:
            self.var_predictor = VariancePredictor(
                embedding_dim,
                var_pred_hidden_dim,
                var_pred_kernel_size,
                var_pred_dropout,
            )
        self.multi_speaker = multi_speaker
        self.normalize_speaker_embeddings = normalize_speaker_embeddings
        self.skip_token_embedding = skip_token_embedding

    @staticmethod
    def _upsample(x, max_frames):
        """
        Upsamples the input tensor to match the specified max_frames.
        """
        batch, hidden_dim, cond_length = x.size()
        x = x.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)
        x = x.view(batch, hidden_dim, max_frames)
        return x

    def forward(self, x, g=None, spk=None):
        """
        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            feature input tensor.
        g : torch.Tensor (batch, 1, time)
            global conditioning input tensor.
        spk : torch.Tensor
            Speaker embeddings

        Returns
        -------
        Generator output
        """
        if self.skip_token_embedding:
            u = x
        else:
            u = self.unit_embedding(x)

        batch_size, time, channel, emb_size = u.shape
        u_ = u.view(batch_size * time, channel, emb_size)

        if self.pooling_type == "attention":
            attn_scores = self.attn_pooling(u_)
            attn_weights = F.softmax(attn_scores, dim=1)
            u_weighted = u_ * attn_weights
            u_pooled = torch.sum(u_weighted, dim=1)
        elif self.pooling_type == "sum":
            u_pooled = torch.sum(u_, dim=1)
        elif self.pooling_type == "none":
            u_pooled = u_

        u = u_pooled.view(batch_size, time, emb_size)
        u = u.transpose(1, 2)

        log_dur = None
        log_dur_pred = None

        if self.duration_predictor:
            uniq_code_feat, uniq_code_mask, dur = process_duration(
                x, u.transpose(1, 2)
            )
            log_dur_pred = self.var_predictor(uniq_code_feat)
            log_dur_pred = log_dur_pred[uniq_code_mask]
            log_dur = torch.log(dur + 1)

        if self.multi_speaker:
            if self.normalize_speaker_embeddings:
                spk = torch.nn.functional.normalize(spk)
            spk = spk.unsqueeze(-1)
            spk = self._upsample(spk, u.shape[-1])
            u = torch.cat([u, spk], dim=1)

        return super().forward(u), (log_dur_pred, log_dur)

    @torch.no_grad()
    def inference(self, x, spk=None):
        """The inference function performs duration prediction and runs the forward method.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            feature input tensor.
        spk : torch.Tensor
            Speaker embeddings

        Returns
        -------
        Generator output
        """
        if not self.skip_token_embedding:
            x = self.unit_embedding(x)

        batch_size, time, channel, emb_size = x.shape
        x_ = x.view(batch_size * time, channel, emb_size)

        if self.pooling_type == "attention":
            attn_scores = self.attn_pooling(x_)
            attn_weights = F.softmax(attn_scores, dim=1)
            x_weighted = x_ * attn_weights
            x_pooled = torch.sum(x_weighted, dim=1)
        elif self.pooling_type == "sum":
            x_pooled = torch.sum(x_, dim=1)
        elif self.pooling_type == "none":
            x_pooled = x_

        x = x_pooled.view(batch_size, time, emb_size)
        x = x.transpose(1, 2)

        if self.duration_predictor:
            assert x.size(0) == 1, (
                "only support single sample batch in inference"
            )
            log_dur_pred = self.var_predictor(x.transpose(1, 2))
            dur_out = torch.clamp(
                torch.round(torch.exp(log_dur_pred) - 1).long(), min=1
            )
            # B x C x T
            x = torch.repeat_interleave(x, dur_out.view(-1), dim=2)

        if self.multi_speaker:
            if self.normalize_speaker_embeddings:
                spk = torch.nn.functional.normalize(spk)
            spk = spk.unsqueeze(-1)
            spk = self._upsample(spk, x.shape[-1])
            x = torch.cat([x, spk], dim=1)

        return super().forward(x)


##################################
# DISCRIMINATOR
##################################


class DiscriminatorP(torch.nn.Module):
    """HiFiGAN Periodic Discriminator
    Takes every Pth value from the input waveform and applies a stack of convolutions.
    Note:
        if period is 2
        waveform = [1, 2, 3, 4, 5, 6 ...] --> [1, 3, 5 ... ] --> convs -> score, feat

    Arguments
    ---------
    period : int
       Take every a new value every `period`
    kernel_size : int
        Size of 1-d kernel for conv stack
    stride : int
        Stride of conv stack
    """

    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList(
            [
                Conv2d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv2d(
                    in_channels=32,
                    out_channels=128,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv2d(
                    in_channels=128,
                    out_channels=512,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv2d(
                    in_channels=512,
                    out_channels=1024,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv2d(
                    in_channels=1024,
                    out_channels=1024,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
            ]
        )
        self.conv_post = Conv2d(
            in_channels=1024,
            out_channels=1,
            kernel_size=(3, 1),
            stride=1,
            padding="same",
            skip_transpose=True,
            weight_norm=True,
        )

    def forward(self, x):
        """
        Arguments
        ---------
        x : torch.Tensor (batch, 1, time)
            input waveform.

        Returns
        -------
        Scores and features
        """

        feat = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            feat.append(x)
        x = self.conv_post(x)
        feat.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feat


class MultiPeriodDiscriminator(torch.nn.Module):
    """HiFiGAN Multi-Period Discriminator (MPD)
    Wrapper for the `PeriodDiscriminator` to apply it in different periods.
    Periods are suggested to be prime numbers to reduce the overlap between each discriminator.
    """

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(2),
                DiscriminatorP(3),
                DiscriminatorP(5),
                DiscriminatorP(7),
                DiscriminatorP(11),
            ]
        )

    def forward(self, x):
        """Returns Multi-Period Discriminator scores and features

        Arguments
        ---------
        x : torch.Tensor (batch, 1, time)
            input waveform.

        Returns
        -------
        Scores and features
        """

        scores = []
        feats = []
        for _, d in enumerate(self.discriminators):
            score, feat = d(x)
            scores.append(score)
            feats.append(feat)
        return scores, feats


class DiscriminatorS(torch.nn.Module):
    """HiFiGAN Scale Discriminator.
    It is similar to `MelganDiscriminator` but with a specific architecture explained in the paper.
    SpeechBrain CNN wrappers are not used here because spectral_norm is not often used

    Arguments
    ---------
    use_spectral_norm : bool
        if `True` switch to spectral norm instead of weight norm.
    """

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = (
            nn.utils.spectral_norm
            if use_spectral_norm
            else nn.utils.weight_norm
        )
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        """
        Arguments
        ---------
        x : torch.Tensor (batch, 1, time)
            input waveform.

        Returns
        -------
        Scores and features
        """

        feat = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            feat.append(x)
        x = self.conv_post(x)
        feat.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feat


class MultiScaleDiscriminator(torch.nn.Module):
    """HiFiGAN Multi-Scale Discriminator.
    Similar to MultiScaleMelganDiscriminator but specially tailored for HiFiGAN as in the paper.
    """

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True),
                DiscriminatorS(),
                DiscriminatorS(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)]
        )

    def forward(self, x):
        """
        Arguments
        ---------
        x : torch.Tensor (batch, 1, time)
            input waveform.

        Returns
        -------
        Scores and features
        """

        scores = []
        feats = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                x = self.meanpools[i - 1](x)
            score, feat = d(x)
            scores.append(score)
            feats.append(feat)
        return scores, feats


class HifiganDiscriminator(nn.Module):
    """HiFiGAN discriminator wrapping MPD and MSD.

    Example
    -------
    >>> inp_tensor = torch.rand([4, 1, 8192])
    >>> hifigan_discriminator = HifiganDiscriminator()
    >>> scores, feats = hifigan_discriminator(inp_tensor)
    >>> len(scores)
    8
    >>> len(feats)
    8

    """

    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(self, x):
        """Returns list of list of features from each layer of each discriminator.

        Arguments
        ---------
        x : torch.Tensor
            input waveform.

        Returns
        -------
        Features from each discriminator layer
        """

        scores, feats = self.mpd(x)
        scores_, feats_ = self.msd(x)
        return scores + scores_, feats + feats_


#################################
# GENERATOR LOSSES
#################################


def stft(x, n_fft, hop_length, win_length, window_fn="hann_window"):
    """computes the Fourier transform of short overlapping windows of the input"""
    o = torch.stft(
        x.squeeze(1),
        n_fft,
        hop_length,
        win_length,
    )
    M = o[:, :, :, 0]
    P = o[:, :, :, 1]
    S = torch.sqrt(torch.clamp(M**2 + P**2, min=1e-8))
    return S


class STFTLoss(nn.Module):
    """STFT loss. Input generate and real waveforms are converted
    to spectrograms compared with L1 and Spectral convergence losses.
    It is from ParallelWaveGAN paper https://arxiv.org/pdf/1910.11480.pdf

    Arguments
    ---------
    n_fft : int
        size of Fourier transform.
    hop_length : int
        the distance between neighboring sliding window frames.
    win_length : int
        the size of window frame and STFT filter.
    """

    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, y_hat, y):
        """Returns magnitude loss and spectral convergence loss

        Arguments
        ---------
        y_hat : torch.tensor
            generated waveform tensor
        y : torch.tensor
            real waveform tensor

        Returns
        -------
        Magnitude loss and spectral convergence loss
        """

        y_hat_M = stft(y_hat, self.n_fft, self.hop_length, self.win_length)
        y_M = stft(y, self.n_fft, self.hop_length, self.win_length)
        # magnitude loss
        loss_mag = F.l1_loss(torch.log(y_M), torch.log(y_hat_M))
        # spectral convergence loss
        loss_sc = torch.norm(y_M - y_hat_M, p="fro") / torch.norm(y_M, p="fro")
        return loss_mag, loss_sc


class MultiScaleSTFTLoss(torch.nn.Module):
    """Multi-scale STFT loss. Input generate and real waveforms are converted
    to spectrograms compared with L1 and Spectral convergence losses.
    It is from ParallelWaveGAN paper https://arxiv.org/pdf/1910.11480.pdf"""

    def __init__(
        self,
        n_ffts=(1024, 2048, 512),
        hop_lengths=(120, 240, 50),
        win_lengths=(600, 1200, 240),
    ):
        super().__init__()
        self.loss_funcs = torch.nn.ModuleList()
        for n_fft, hop_length, win_length in zip(
            n_ffts, hop_lengths, win_lengths
        ):
            self.loss_funcs.append(STFTLoss(n_fft, hop_length, win_length))

    def forward(self, y_hat, y):
        """Returns multi-scale magnitude loss and spectral convergence loss

        Arguments
        ---------
        y_hat : torch.tensor
            generated waveform tensor
        y : torch.tensor
            real waveform tensor

        Returns
        -------
        Magnitude loss and spectral convergence loss
        """

        N = len(self.loss_funcs)
        loss_sc = 0
        loss_mag = 0
        for f in self.loss_funcs:
            lm, lsc = f(y_hat, y)
            loss_mag += lm
            loss_sc += lsc
        loss_sc /= N
        loss_mag /= N
        return loss_mag, loss_sc


class L1SpecLoss(nn.Module):
    """L1 Loss over Spectrograms as described in HiFiGAN paper https://arxiv.org/pdf/2010.05646.pdf
    Note : L1 loss helps leaning details compared with L2 loss

    Arguments
    ---------
    sample_rate : int
        Sample rate of audio signal.
    hop_length : int
        Length of hop between STFT windows.
    win_length : int
        Window size.
    n_mel_channels : int
        Number of mel filterbanks.
    n_fft : int
        Size of FFT.
    n_stft : int
        Size of STFT.
    mel_fmin : float
        Minimum frequency.
    mel_fmax : float
        Maximum frequency.
    mel_normalized : bool
        Whether to normalize by magnitude after stft.
    power : float
        Exponent for the magnitude spectrogram.
    norm : str or None
        If "slaney", divide the triangular mel weights by the width of the mel band
    mel_scale : str
        Scale to use: "htk" or "slaney".
    dynamic_range_compression : bool
        whether to do dynamic range compression
    """

    def __init__(
        self,
        sample_rate=22050,
        hop_length=256,
        win_length=24,
        n_mel_channels=80,
        n_fft=1024,
        n_stft=1024 // 2 + 1,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        mel_normalized=False,
        power=1.0,
        norm="slaney",
        mel_scale="slaney",
        dynamic_range_compression=True,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.n_fft = n_fft
        self.n_stft = n_fft // 2 + 1
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.mel_normalized = mel_normalized
        self.power = power
        self.norm = norm
        self.mel_scale = mel_scale
        self.dynamic_range_compression = dynamic_range_compression

    def forward(self, y_hat, y):
        """Returns L1 Loss over Spectrograms

        Arguments
        ---------
        y_hat : torch.tensor
            generated waveform tensor
        y : torch.tensor
            real waveform tensor

        Returns
        -------
        L1 loss
        """
        y_hat_M = mel_spectogram(
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.n_fft,
            self.n_mel_channels,
            self.mel_fmin,
            self.mel_fmax,
            self.power,
            self.mel_normalized,
            self.norm,
            self.mel_scale,
            self.dynamic_range_compression,
            y_hat,
        )
        # y_M = mel_spectogram(self.mel_params, y)
        y_M = mel_spectogram(
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.n_fft,
            self.n_mel_channels,
            self.mel_fmin,
            self.mel_fmax,
            self.power,
            self.mel_normalized,
            self.norm,
            self.mel_scale,
            self.dynamic_range_compression,
            y,
        )

        # magnitude loss
        # loss_mag = F.l1_loss(torch.log(y_M), torch.log(y_hat_M))
        loss_mag = F.l1_loss(y_M, y_hat_M)
        return loss_mag


class MSEGLoss(nn.Module):
    """Mean Squared Generator Loss
    The generator is trained to fake the discriminator by updating the sample quality
    to be classified to a value almost equal to 1.
    """

    def forward(self, score_fake):
        """Returns Generator GAN loss

        Arguments
        ---------
        score_fake : list
            discriminator scores of generated waveforms D(G(s))

        Returns
        -------
        Generator loss
        """

        loss_fake = F.mse_loss(
            score_fake, score_fake.new_ones(score_fake.shape)
        )
        return loss_fake


class HingeGLoss(nn.Module):
    """Hinge Generator Loss.

    The generator is trained to fake the discriminator by updating the sample quality
    to be classified to a value almost equal to 1.

    Example
    -------
    > import torch
    > score_fake = torch.randn(4, 88)
    > loss = HingeGLoss()(score_fake)
    > print(loss)

    """

    def forward(self, score_fake):
        """Returns Generator GAN loss

        Arguments
        ---------
        score_fake : torch.Tensor
            Discriminator scores of generated waveforms D(G(s))

        Returns
        -------
        Generator loss
        """
        loss_fake = (1 - score_fake).clamp(min=0).mean()
        return loss_fake


class MelganFeatureLoss(nn.Module):
    """Calculates the feature matching loss, which is a learned similarity metric measured by
    the difference in features of the discriminator between a ground truth sample and a generated
    sample (Larsen et al., 2016, Kumar et al., 2019).
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.loss_func = nn.L1Loss()

    # pylint: disable=no-self-use
    def forward(self, fake_feats, real_feats):
        """Returns feature matching loss

        Arguments
        ---------
        fake_feats : list
            discriminator features of generated waveforms
        real_feats : list
            discriminator features of groundtruth waveforms

        Returns
        -------
        Feature matching loss
        """

        loss_feats = 0
        num_feats = 0
        for idx, _ in enumerate(fake_feats):
            for fake_feat, real_feat in zip(fake_feats[idx], real_feats[idx]):
                loss_feats += self.loss_func(fake_feat, real_feat)
                num_feats += 1
        loss_feats = loss_feats / num_feats
        return loss_feats


##################################
# DISCRIMINATOR LOSSES
##################################


class MSEDLoss(nn.Module):
    """Mean Squared Discriminator Loss
    The discriminator is trained to classify ground truth samples to 1,
    and the samples synthesized from the generator to 0.
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.loss_func = nn.MSELoss()

    def forward(self, score_fake, score_real):
        """Returns Discriminator GAN losses

        Arguments
        ---------
        score_fake : list
            discriminator scores of generated waveforms
        score_real : list
            discriminator scores of groundtruth waveforms

        Returns
        -------
        Discriminator losses
        """

        loss_real = self.loss_func(
            score_real, score_real.new_ones(score_real.shape)
        )
        loss_fake = self.loss_func(
            score_fake, score_fake.new_zeros(score_fake.shape)
        )
        loss_d = loss_real + loss_fake
        return loss_d, loss_real, loss_fake


class HingeDLoss(nn.Module):
    """Hinge Discriminator Loss.

    The discriminator is trained to classify ground truth samples to 1,
    and the samples synthesized from the generator to 0.

    Example
    -------
    > import torch
    > score_fake = torch.randn(4, 88)
    > score_real = torch.randn(4, 88)
    > loss = HingeDLoss()(score_fake, score_real)
    > print(loss)

    """

    def forward(self, score_fake, score_real):
        """Returns Discriminator GAN losses

        Arguments
        ---------
        score_fake : torch.Tensor
            discriminator scores of generated waveforms
        score_real : torch.Tensor
            discriminator scores of groundtruth waveforms

        Returns
        -------
        Discriminator losses
        """
        loss_real = (1 - score_real).clamp(min=0).mean()
        loss_fake = (1 + score_fake).clamp(min=0).mean()
        loss_d = loss_real + loss_fake
        return loss_d, loss_real, loss_fake


#####################################
# LOSS WRAPPERS
#####################################


def _apply_G_adv_loss(scores_fake, loss_func):
    """Compute Generator adversarial loss function
    and normalize values

    Arguments
    ---------
    scores_fake : list
        discriminator scores of generated waveforms
    loss_func : object
        object of target generator loss

    Returns
    -------
    Generator loss
    """

    adv_loss = 0
    if isinstance(scores_fake, list):
        for score_fake in scores_fake:
            fake_loss = loss_func(score_fake)
            adv_loss += fake_loss
        # adv_loss /= len(scores_fake)
    else:
        fake_loss = loss_func(scores_fake)
        adv_loss = fake_loss
    return adv_loss


def _apply_D_loss(scores_fake, scores_real, loss_func):
    """Compute Discriminator losses and normalize loss values

    Arguments
    ---------
    scores_fake : list
        discriminator scores of generated waveforms
    scores_real : list
        discriminator scores of groundtruth waveforms
    loss_func : object
        object of target discriminator loss

    Returns
    -------
    Discriminator losses
    """

    loss = 0
    real_loss = 0
    fake_loss = 0
    if isinstance(scores_fake, list):
        # multi-scale loss
        for score_fake, score_real in zip(scores_fake, scores_real):
            total_loss, real_loss, fake_loss = loss_func(
                score_fake=score_fake, score_real=score_real
            )
            loss += total_loss
            real_loss += real_loss
            fake_loss += fake_loss
        # normalize loss values with number of scales (discriminators)
        # loss /= len(scores_fake)
        # real_loss /= len(scores_real)
        # fake_loss /= len(scores_fake)
    else:
        # single scale loss
        total_loss, real_loss, fake_loss = loss_func(scores_fake, scores_real)
        loss = total_loss
    return loss, real_loss, fake_loss


##################################
# MODEL LOSSES
##################################


class GeneratorLoss(nn.Module):
    """Creates a summary of generator losses
    and applies weights for different losses

    Arguments
    ---------
    stft_loss : object
        object of stft loss
    stft_loss_weight : float
        weight of STFT loss
    mseg_loss : object
        object of mseg loss
    mseg_loss_weight : float
        weight of mseg loss
    feat_match_loss : object
        object of feature match loss
    feat_match_loss_weight : float
        weight of feature match loss
    l1_spec_loss : object
        object of L1 spectrogram loss
    l1_spec_loss_weight : float
        weight of L1 spectrogram loss
    mseg_dur_loss : object
        object of mseg duration loss
    mseg_dur_loss_weight : float
        weight of mseg duration loss
    """

    def __init__(
        self,
        stft_loss=None,
        stft_loss_weight=0,
        mseg_loss=None,
        mseg_loss_weight=0,
        feat_match_loss=None,
        feat_match_loss_weight=0,
        l1_spec_loss=None,
        l1_spec_loss_weight=0,
        mseg_dur_loss=None,
        mseg_dur_loss_weight=0,
    ):
        super().__init__()
        self.stft_loss = stft_loss
        self.stft_loss_weight = stft_loss_weight
        self.mseg_loss = mseg_loss
        self.mseg_loss_weight = mseg_loss_weight
        self.feat_match_loss = feat_match_loss
        self.feat_match_loss_weight = feat_match_loss_weight
        self.l1_spec_loss = l1_spec_loss
        self.l1_spec_loss_weight = l1_spec_loss_weight
        self.mseg_dur_loss = mseg_dur_loss
        self.mseg_dur_loss_weight = mseg_dur_loss_weight

    def forward(
        self,
        stage,
        y_hat=None,
        y=None,
        scores_fake=None,
        feats_fake=None,
        feats_real=None,
        log_dur_pred=None,
        log_dur=None,
    ):
        """Returns a dictionary of generator losses and applies weights

        Arguments
        ---------
        stage : speechbrain.Stage
            training, validation or testing
        y_hat : torch.tensor
            generated waveform tensor
        y : torch.tensor
            real waveform tensor
        scores_fake : list
            discriminator scores of generated waveforms
        feats_fake : list
            discriminator features of generated waveforms
        feats_real : list
            discriminator features of groundtruth waveforms
        log_dur_pred : torch.Tensor
            Predicted duration for duration loss
        log_dur : torch.Tensor
            Real duration for duration loss

        Returns
        -------
        Dictionary of generator losses
        """

        gen_loss = 0
        adv_loss = 0
        dur_loss = 0
        loss = {}

        # STFT Loss
        if self.stft_loss:
            stft_loss_mg, stft_loss_sc = self.stft_loss(
                y_hat[:, :, : y.size(2)].squeeze(1), y.squeeze(1)
            )
            loss["G_stft_loss_mg"] = stft_loss_mg
            loss["G_stft_loss_sc"] = stft_loss_sc
            gen_loss = gen_loss + self.stft_loss_weight * (
                stft_loss_mg + stft_loss_sc
            )

        # L1 Spec loss
        if self.l1_spec_loss:
            l1_spec_loss = self.l1_spec_loss(y_hat, y)
            loss["G_l1_spec_loss"] = l1_spec_loss
            gen_loss = gen_loss + self.l1_spec_loss_weight * l1_spec_loss

        # multiscale MSE adversarial loss
        if self.mseg_loss and scores_fake is not None:
            mse_fake_loss = _apply_G_adv_loss(scores_fake, self.mseg_loss)
            loss["G_mse_fake_loss"] = mse_fake_loss
            adv_loss = adv_loss + self.mseg_loss_weight * mse_fake_loss

        # Feature Matching Loss
        if self.feat_match_loss and feats_fake is not None:
            feat_match_loss = self.feat_match_loss(feats_fake, feats_real)
            loss["G_feat_match_loss"] = feat_match_loss
            adv_loss = adv_loss + self.feat_match_loss_weight * feat_match_loss

        # Duration loss
        if self.mseg_dur_loss and stage == sb.Stage.TRAIN:
            dur_loss = F.mse_loss(log_dur_pred, log_dur, reduction="mean")
            loss["G_dur_loss"] = dur_loss
            dur_loss *= self.mseg_dur_loss_weight

        loss["G_loss"] = gen_loss + adv_loss + dur_loss
        loss["G_gen_loss"] = gen_loss
        loss["G_adv_loss"] = adv_loss

        return loss


class DiscriminatorLoss(nn.Module):
    """Creates a summary of discriminator losses

    Arguments
    ---------
    msed_loss : object
        object of MSE discriminator loss
    """

    def __init__(self, msed_loss=None):
        super().__init__()
        self.msed_loss = msed_loss

    def forward(self, scores_fake, scores_real):
        """Returns a dictionary of discriminator losses

        Arguments
        ---------
        scores_fake : list
            discriminator scores of generated waveforms
        scores_real : list
            discriminator scores of groundtruth waveforms

        Returns
        -------
        Dictionary of discriminator losses
        """

        disc_loss = 0
        loss = {}

        if self.msed_loss:
            mse_D_loss, mse_D_real_loss, mse_D_fake_loss = _apply_D_loss(
                scores_fake=scores_fake,
                scores_real=scores_real,
                loss_func=self.msed_loss,
            )
            loss["D_mse_gan_loss"] = mse_D_loss
            loss["D_mse_gan_real_loss"] = mse_D_real_loss
            loss["D_mse_gan_fake_loss"] = mse_D_fake_loss
            disc_loss += mse_D_loss

        loss["D_loss"] = disc_loss
        return loss

