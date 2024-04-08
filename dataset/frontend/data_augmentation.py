# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.09
""" Implementation of data augmention
    Modified from wenet/dataset/processor.py
"""

import random
import torch
import torchaudio


class NoiseProcessor(object):
    """ Interface of unitities of add_noise, Borrowed from 
        nemo.collections.asr.parts.preprocessing.segment 
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def rms_db(pcm: torch.Tensor):
        # pcm is torch.Tensor get through torchaudio.load
        mean_square = (pcm**2).mean()
        return 10 * torch.log10(mean_square)

    @staticmethod
    def gain_db(pcm, gain):
        pcm *= 10.0**(gain / 20.0)
        return pcm


def add_noise(pcm: torch.Tensor,
              noise_pcm: torch.Tensor,
              min_snr_db=10,
              max_snr_db=50,
              max_gain_db=300.0):
    """ Add noise if 'noise_pcm' field provided, the ratio of noise augment control
        has been relocated to outside of func
    """
    snr_db = random.uniform(min_snr_db, max_snr_db)
    data_rms = NoiseProcessor.rms_db(pcm)
    noise_rms = NoiseProcessor.rms_db(noise_pcm)
    noise_gain_db = min(data_rms - noise_rms - snr_db, max_gain_db)

    noise_pcm = NoiseProcessor.gain_db(noise_pcm, noise_gain_db)
    if pcm.shape[1] > noise_pcm.shape[1]:
        noise_pcm = noise_pcm.repeat(
            1,
            torch.div(pcm.shape[1], noise_pcm.shape[1], rounding_mode="floor") +
            1)
    return pcm + noise_pcm[:, :pcm.shape[1]]


def speed_perturb(pcm: torch.Tensor,
                  sample_rate=16000,
                  min_speed=0.9,
                  max_speed=1.1,
                  rate=3):
    """ Apply speed perturb to the data.
        Inplace operation.
    """
    speeds = torch.linspace(min_speed, max_speed, steps=rate).tolist()
    speed = random.choice(speeds)
    if speed != 1.0:
        perturbed_pcm, _ = torchaudio.sox_effects.apply_effects_tensor(
            pcm, sample_rate,
            [['speed', str(speed)], ['rate', str(sample_rate)]])
        return perturbed_pcm
    else:
        return pcm


def volume_perturb(pcm: torch.Tensor, min_gain=1, max_gain=1) -> torch.Tensor:
    """ Multiply the pcms with a gain perturbing volume (range from min_gain to max_gain) """
    gain = random.uniform(min_gain, max_gain)
    pcm_auged = pcm * gain
    pcm_auged = torch.clamp(pcm_auged, min=-1,
                            max=1)  # Clamp since pcm shall be normed.
    return pcm_auged
