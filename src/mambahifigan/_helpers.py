"""Helpers the generators share, as they appear in the project's HifiGAN.py."""

import torch


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
