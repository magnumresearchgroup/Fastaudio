from __future__ import division
import torch
import numpy as np
import math
import logging
from packaging import version
from nnAudio.Spectrogram import *

from speechbrain.processing.features import (
    STFT,
    spectral_magnitude,
    Filterbank,
    Deltas,
    ContextWindow,
)
from speechbrain.utils.checkpoints import (
    mark_as_saver,
    mark_as_loader,
    mark_as_transfer,
    register_checkpoint_hooks,
)

from leaf_audio_pytorch import postprocessing
from torch import nn
from torch.nn import functional as F

class mag(torch.nn.Module):
    def __init__(
            self,
            deltas=False,
            context=False,
            requires_grad=False,
            sample_rate=16000,
            f_min=0,
            f_max=None,
            n_fft=400,
            n_mels=40,
            filter_shape="triangular",
            param_change_factor=1.0,
            param_rand_factor=0.0,
            left_frames=5,
            right_frames=5,
            win_length=25,
            hop_length=10,
    ):
        super().__init__()
        self.deltas = deltas
        self.context = context
        self.requires_grad = requires_grad

        if f_max is None:
            f_max = sample_rate / 2

        self.compute_STFT = STFT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )

        self.compute_deltas = Deltas(input_size=n_mels)
        self.context_window = ContextWindow(
            left_frames=left_frames, right_frames=right_frames,
        )

    def forward(self, wav):
        """Returns a set of features generated from the input waveforms.

        Arguments
        ---------
        wav : tensor
            A batch of audio signals to transform to features.
        """
        with torch.no_grad():

            STFT = self.compute_STFT(wav)
            mag = spectral_magnitude(STFT, log=True)

        return mag

class IF(torch.nn.Module):
    def __init__(
        self,
        deltas=False,
        context=False,
        requires_grad=False,
        sample_rate=16000,
        f_min=0,
        f_max=None,
        n_fft=400,
        n_mels=40,
        filter_shape="triangular",
        param_change_factor=1.0,
        param_rand_factor=0.0,
        left_frames=5,
        right_frames=5,
        win_length=25,
        hop_length=10,
    ):
        super().__init__()
        self.deltas = deltas
        self.context = context
        self.requires_grad = requires_grad

        if f_max is None:
            f_max = sample_rate / 2

        self.compute_STFT = STFT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )
        self.compute_fbanks = Filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            freeze=not requires_grad,
            filter_shape=filter_shape,
            param_change_factor=param_change_factor,
            param_rand_factor=param_rand_factor,
        )
        self.compute_deltas = Deltas(input_size=n_mels)
        self.context_window = ContextWindow(
            left_frames=left_frames, right_frames=right_frames,
        )


    def forward(self, wav):
        def diff(x, axis):
            shape = x.shape
            begin_back = [0 for unused_s in range(len(shape))]
            begin_front = [0 for unused_s in range(len(shape))]
            begin_front[axis] = 1
            size = list(shape)
            size[axis] -= 1
            slice_front = x[begin_front[0]:begin_front[0] + size[0], begin_front[1]:begin_front[1] + size[1]]
            slice_back = x[begin_back[0]:begin_back[0] + size[0], begin_back[1]:begin_back[1] + size[1]]
            d = slice_front - slice_back
            return d


        def unwrap(p, discont=np.pi, axis=-1):
            dd = diff(p, axis=axis)
            ddmod = np.mod(dd + np.pi, 2.0 * np.pi) - np.pi  # ddmod = tf.mod(dd + np.pi, 2.0 * np.pi) - np.pi
            idx = np.logical_and(np.equal(ddmod, -np.pi),
                                np.greater(dd, 0))  # idx = tf.logical_and(tf.equal(ddmod, -np.pi), tf.greater(dd, 0))
            ddmod = np.where(idx, np.ones_like(ddmod) * np.pi,
                            ddmod)  # ddmod = tf.where(idx, tf.ones_like(ddmod) * np.pi, ddmod)
            ph_correct = ddmod - dd
            idx = np.less(np.abs(dd), discont)  # idx = tf.less(tf.abs(dd), discont)
            ddmod = np.where(idx, np.zeros_like(ddmod), dd)  # ddmod = tf.where(idx, tf.zeros_like(ddmod), dd)
            ph_cumsum = np.cumsum(ph_correct, axis=axis)  # ph_cumsum = tf.cumsum(ph_correct, axis=axis)
            shape = np.array(p.shape)  # shape = p.get_shape().as_list()
            shape[axis] = 1
            ph_cumsum = np.concatenate([np.zeros(shape, dtype=p.dtype), ph_cumsum], axis=axis)
            unwrapped = p + ph_cumsum
            return unwrapped


        def instantaneous_frequency(phase_angle, time_axis):
            phase_unwrapped = unwrap(phase_angle, axis=time_axis)
            dphase = diff(phase_unwrapped, axis=time_axis)
            size = np.array(phase_unwrapped.shape)
            size[time_axis] = 1
            begin = [0 for unused_s in size]
            phase_slice = phase_unwrapped[begin[0]:begin[0] + size[0], begin[1]:begin[1] + size[1]]
            dphase = np.concatenate([phase_slice, dphase], axis=time_axis) / np.pi
            return dphase

        with torch.no_grad():
            # stft = CQT(sr=16000, verbose=False)
            # X = stft(wav)
            # X=torch.einsum('ijk->ikj', X)
            D= torch.stft(wav, n_fft=512, window=torch.hann_window(512).cuda(), return_complex=True)
            phase=torch.angle(D)
            phase=torch.einsum('ijk->ikj', phase)
            IF=torch.tensor(instantaneous_frequency(phase.cpu().numpy(), -2))
        return IF.cuda()

class Gabor(object):
        def __init__(self,
                    nfilters=40,
                    min_freq=0,
                    max_freq=8000,
                    fs=16000,
                    wlen=25,
                    wstride=10,
                    nfft=512,
                    normalize_energy=False):
                if not nfilters > 0:
                    raise(Exception,
                    'Number of filters must be positive, not {0:%d}'.format(nfilters))
                if max_freq > fs // 2:
                    raise(Exception,
                    'Upper frequency %f exceeds Nyquist %f' % (max_freq, fs // 2))
                self.nfilters = nfilters
                self.min_freq = min_freq
                self.max_freq = max_freq
                self.fs = fs
                self.wlen = wlen
                self.wstride = wstride
                self.nfft = nfft
                self.normalize_energy = normalize_energy
                self._build_mels()
                self._build_gabors()

        def _hz2mel(self, f):
            # Converts a frequency in hertz to mel
            return 2595 * np.log10(1+f/700)

        def _mel2hz(self, m):
            # Converts a frequency in mel to hertz
            return 700 * (np.power(10, m/2595) - 1)

        def _gabor_wavelet(self, eta, sigma):
            T = self.wlen * self.fs / 1000
            # Returns a gabor wavelet on a window of size T

            def gabor_function(t):
                return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(1j * eta * t) * np.exp(-t**2/(2 * sigma**2))
            return np.asarray([gabor_function(t) for t in np.arange(-T/2,T/2 + 1)])

        def _gabor_params_from_mel(self, mel_filter):
            # Parameters in radians
            coeff = np.sqrt(2*np.log(2))*self.nfft
            mel_filter = np.sqrt(mel_filter)
            center_frequency = np.argmax(mel_filter)
            peak = mel_filter[center_frequency]
            half_magnitude = peak/2.0
            spread = np.where(mel_filter >= half_magnitude)[0]
            width = max(spread[-1] - spread[0],1)
            return center_frequency*2*np.pi/self.nfft, coeff/(np.pi*width)

        def _melfilter_energy(self, mel_filter):
            # Computes the energy of a mel-filter (area under the magnitude spectrum)
            height = max(mel_filter)
            hz_spread = (len(np.where(mel_filter > 0)[0])+2)*2*np.pi/self.nfft
            return 0.5 * height * hz_spread

        def _build_mels(self):
            # build mel filter matrix
            self.melfilters = [np.zeros(self.nfft//2 + 1) for i in range(self.nfilters)]
            dfreq = self.fs / self.nfft

            melmax = self._hz2mel(self.max_freq)
            melmin = self._hz2mel(self.min_freq)
            dmelbw = (melmax - melmin) / (self.nfilters + 1)
            # filter edges in hz
            filt_edge = self._mel2hz(melmin + dmelbw *
                                    np.arange(self.nfilters + 2, dtype='d'))
            self.filt_edge = filt_edge
            for filter_idx in range(0, self.nfilters):
                # Filter triangles in dft points
                leftfr = min(round(filt_edge[filter_idx] / dfreq), self.nfft//2)
                centerfr = min(round(filt_edge[filter_idx + 1] / dfreq), self.nfft//2)
                rightfr = min(round(filt_edge[filter_idx + 2] / dfreq), self.nfft//2)
                height = 1
                if centerfr != leftfr:
                    leftslope = height / (centerfr - leftfr)
                else:
                    leftslope = 0
                freq = leftfr + 1
                while freq < centerfr:
                    self.melfilters[filter_idx][int(freq)] = (freq - leftfr) * leftslope
                    freq += 1
                if freq == centerfr:
                    self.melfilters[filter_idx][int(freq)] = height
                    freq += 1
                if centerfr != rightfr:
                    rightslope = height / (centerfr - rightfr)
                while freq < rightfr:
                    self.melfilters[filter_idx][int(freq)] = (freq - rightfr) * rightslope
                    freq += 1
                if self.normalize_energy:
                    energy = self._melfilter_energy(self.melfilters[filter_idx])
                    self.melfilters[filter_idx] /= energy

        def _build_gabors(self):
            self.gaborfilters = []
            self.sigmas = []
            self.center_frequencies = []
            for mel_filter in self.melfilters:
                center_frequency, sigma = self._gabor_params_from_mel(mel_filter)
                self.sigmas.append(sigma)
                self.center_frequencies.append(center_frequency)
                gabor_filter = self._gabor_wavelet(center_frequency, sigma)
                # Renormalize the gabor wavelets
                gabor_filter = gabor_filter * np.sqrt(self._melfilter_energy(mel_filter)*2*np.sqrt(np.pi)*sigma)
                self.gaborfilters.append(gabor_filter)




class TDFbanks(nn.Module):
    
    def __init__(self,
                 mode,
                 nfilters,
                 samplerate=16000,
                 wlen=25,
                 wstride=10,
                 compression='log',
                 preemp=False,
                 mvn=False,
                 min_freq=0,
                max_freq=8000,
                nfft=512,
                window_type='hanning',
                normalize_energy=False,
                alpha=0.97):
        super().__init__()
        def chirp(f0, f1, T, fs):
    # f0 is the lower bound of the frequency range, in Hz
    # f1 is the upper bound of the frequency range, in Hz
    # T is the duration of the chirp, in seconds
    # fs is the sampling rate
            slope = (f1-f0)/float(T)

            def chirp_wave(t):
                return np.cos((0.5*slope*t+f0)*2*np.pi*t)
            return [chirp_wave(t) for t in np.linspace(0, T, T*fs).tolist()]


        def window(window_type, N):
            def hanning(n):
                return 0.5*(1 - np.cos(2 * np.pi * (n - 1) / (N - 1)))

            def hamming(n):
                return 0.54 - 0.46 * np.cos(2 * np.pi * (n - 1) / (N - 1))

            if window_type == 'hanning':
                return np.asarray([hanning(n) for n in range(N)])
            else:
                return np.asarray([hamming(n) for n in range(N)])

        window_size = samplerate * wlen // 1000 + 1
        window_stride = samplerate * wstride // 1000
        padding_size = (window_size - 1) // 2
        self.preemp = None
        if preemp:
            self.preemp = nn.Conv1d(1, 1, 2, 1, padding=1, groups=1, bias=False)
        self.complex_conv = nn.Conv1d(1, 2 * nfilters, window_size, 1,
            padding=padding_size, groups=1, bias=False)
        self.modulus = nn.LPPool1d(2, 2, stride=2)
        self.lowpass = nn.Conv1d(nfilters, nfilters, window_size, window_stride,
            padding=0, groups=nfilters, bias=False)
        if mode == 'Fixed':
            for param in self.parameters():
                param.requires_grad = False
        elif mode == 'learnfbanks':
            if preemp:
                self.preemp.weight.requires_grad = False
            self.lowpass.weight.requires_grad = False
        if mvn:
            self.instancenorm = nn.InstanceNorm1d(nfilters, momentum=1)
        self.nfilters = nfilters
        self.fs = samplerate
        self.wlen = wlen
        self.wstride = wstride
        self.compression = compression
        self.mvn = mvn
        if self.preemp:
            self.preemp.weight.data[0][0][0] = -alpha
            self.preemp.weight.data[0][0][1] = 1
        # Initialize complex convolution
        self.complex_init = Gabor(self.nfilters,
                                             min_freq,
                                             max_freq,
                                             self.fs,
                                             self.wlen,
                                             self.wstride,
                                             nfft,
                                             normalize_energy)
        for idx, gabor in enumerate(self.complex_init.gaborfilters):
            self.complex_conv.weight.data[2*idx][0].copy_(
                torch.from_numpy(np.real(gabor)))
            self.complex_conv.weight.data[2*idx + 1][0].copy_(
                torch.from_numpy(np.imag(gabor)))
        # Initialize lowpass
        self.lowpass_init = window(window_type,
                                         (self.fs * self.wlen)//1000 + 1)
        for idx in range(self.nfilters):
            self.lowpass.weight.data[idx][0].copy_(
                torch.from_numpy(self.lowpass_init))

        

    def forward(self, x):
        # Reshape waveform to format (1,1,seq_length)
        # x = x.view(1, 1, -1)
        # Preemphasis
        if self.preemp:
            x = self.preemp(x)
        # Complex convolution
        x = self.complex_conv(x)
        # Squared modulus operator
        x = x.transpose(1, 2)
        x = F.avg_pool1d(x.pow(2), 2, 2, 0, False).mul(2)
        x = x.transpose(1, 2)
        x = self.lowpass(x)
        x = x.abs()
        x = x + 1
        if self.compression == 'log':
            x = x.log()
        # The dimension of x is 1, n_channels, seq_length
        if self.mvn:
            x = self.instancenorm(x)
        return x

    def chirp(f0, f1, T, fs):
    # f0 is the lower bound of the frequency range, in Hz
    # f1 is the upper bound of the frequency range, in Hz
    # T is the duration of the chirp, in seconds
    # fs is the sampling rate
        slope = (f1-f0)/float(T)

        def chirp_wave(t):
            return np.cos((0.5*slope*t+f0)*2*np.pi*t)
        return [chirp_wave(t) for t in np.linspace(0, T, T*fs).tolist()]


    def window(window_type, N):
        def hanning(n):
            return 0.5*(1 - np.cos(2 * np.pi * (n - 1) / (N - 1)))

        def hamming(n):
            return 0.54 - 0.46 * np.cos(2 * np.pi * (n - 1) / (N - 1))

        if window_type == 'hanning':
            return np.asarray([hanning(n) for n in range(N)])
        else:
            return np.asarray([hamming(n) for n in range(N)])


    