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




class Filterbank(torch.nn.Module):
    """computes filter bank (FBANK) features given spectral magnitudes.

    Arguments
    ---------
    n_mels : float
        Number of Mel filters used to average the spectrogram.
    log_mel : bool
        If True, it computes the log of the FBANKs.
    filter_shape : str
        Shape of the filters ('triangular', 'rectangular', 'gaussian').
    f_min : int
        Lowest frequency for the Mel filters.
    f_max : int
        Highest frequency for the Mel filters.
    n_fft : int
        Number of fft points of the STFT. It defines the frequency resolution
        (n_fft should be<= than win_len).
    sample_rate : int
        Sample rate of the input audio signal (e.g, 16000)
    power_spectrogram : float
        Exponent used for spectrogram computation.
    amin : float
        Minimum amplitude (used for numerical stability).
    ref_value : float
        Reference value used for the dB scale.
    top_db : float
        Top dB valu used for log-mels.
    freeze : bool
        If False, it the central frequency and the band of each filter are
        added into nn.parameters. If True, the standard frozen features
        are computed.
    param_change_factor: bool
        If freeze=False, this parameter affects the speed at which the filter
        parameters (i.e., central_freqs and bands) can be changed.  When high
        (e.g., param_change_factor=1) the filters change a lot during training.
        When low (e.g. param_change_factor=0.1) the filter parameters are more
        stable during training
    param_rand_factor: float
        This parameter can be used to randomly change the filter parameters
        (i.e, central frequencies and bands) during training.  It is thus a
        sort of regularization. param_rand_factor=0 does not affect, while
        param_rand_factor=0.15 allows random variations within +-15% of the
        standard values of the filter parameters (e.g., if the central freq
        is 100 Hz, we can randomly change it from 85 Hz to 115 Hz).

    Example
    -------
    >>> import torch
    >>> compute_fbanks = Filterbank()
    >>> inputs = torch.randn([10, 101, 201])
    >>> features = compute_fbanks(inputs)
    >>> features.shape
    torch.Size([10, 101, 40])
    """

    def __init__(
        self,
        n_mels=40,
        log_mel=True,
        filter_shape="triangular",
        f_min=0,
        f_max=8000,
        n_fft=400,
        sample_rate=16000,
        power_spectrogram=2,
        amin=1e-10,
        ref_value=1.0,
        top_db=80.0,
        param_change_factor=1.0,
        param_rand_factor=0.0,
        freeze=True,
        sort=False
    ):
        super().__init__()
        self.sort=sort
        self.n_mels = n_mels
        self.log_mel = log_mel
        self.filter_shape = filter_shape
        self.f_min = f_min
        self.f_max = f_max
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.power_spectrogram = power_spectrogram
        self.amin = amin
        self.ref_value = ref_value
        self.top_db = top_db
        self.freeze = freeze
        self.n_stft = self.n_fft // 2 + 1
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))
        self.device_inp = torch.device("cpu")
        self.param_change_factor = param_change_factor
        self.param_rand_factor = param_rand_factor

        if self.power_spectrogram == 2:
            self.multiplier = 10
        else:
            self.multiplier = 20

        # Make sure f_min < f_max
        if self.f_min >= self.f_max:
            err_msg = "Require f_min: %f < f_max: %f" % (
                self.f_min,
                self.f_max,
            )
            logger.error(err_msg, exc_info=True)

        # Filter definition
        mel = torch.linspace(
            self._to_mel(self.f_min), self._to_mel(self.f_max), self.n_mels + 2
        )
        hz = self._to_hz(mel)

        # Computation of the filter bands
        band = hz[1:] - hz[:-1]
        self.band = band[:-1]
        self.f_central = hz[1:-1]

        # Adding the central frequency and the band to the list of nn param
        if not self.freeze:
            self.f_central = torch.nn.Parameter(
                self.f_central / (self.sample_rate * self.param_change_factor)
            )
            self.band = torch.nn.Parameter(
                self.band / (self.sample_rate * self.param_change_factor)
            )

        # Frequency axis
        all_freqs = torch.linspace(0, self.sample_rate // 2, self.n_stft)

        # Replicating for all the filters
        self.all_freqs_mat = all_freqs.repeat(self.f_central.shape[0], 1)

    def forward(self, spectrogram):
        """Returns the FBANks.

        Arguments
        ---------
        x : tensor
            A batch of spectrogram tensors.
        """
        f_central=torch.clamp(self.f_central, 0, 0.5)		
        band=torch.clamp(self.band, 3.1/16000, 603.7/16000)		
        if self.sort:		
            f_central, _ = torch.sort(f_central)		
            band, _ = torch.sort(band)
        # Computing central frequency and bandwidth of each filter
        f_central_mat = f_central.repeat(
            self.all_freqs_mat.shape[1], 1
        ).transpose(0, 1)
        band_mat = band.repeat(self.all_freqs_mat.shape[1], 1).transpose(
            0, 1
        )

        # Uncomment to print filter parameters
        # print(self.f_central*self.sample_rate * self.param_change_factor)
        # print(self.band*self.sample_rate* self.param_change_factor)

        # Creation of the multiplication matrix. It is used to create
        # the filters that average the computed spectrogram.
        if not self.freeze:
            f_central_mat = f_central_mat * (
                self.sample_rate
                * self.param_change_factor
                * self.param_change_factor
            )
            band_mat = band_mat * (
                self.sample_rate
                * self.param_change_factor
                * self.param_change_factor
            )

        # Regularization with random changes of filter central frequency and band
        elif self.param_rand_factor != 0 and self.training:
            rand_change = (
                1.0
                + torch.rand(2) * 2 * self.param_rand_factor
                - self.param_rand_factor
            )
            f_central_mat = f_central_mat * rand_change[0]
            band_mat = band_mat * rand_change[1]

        fbank_matrix = self._create_fbank_matrix(f_central_mat, band_mat).to(
            spectrogram.device
        )

        sp_shape = spectrogram.shape

        # Managing multi-channels case (batch, time, channels)
        if len(sp_shape) == 4:
            spectrogram = spectrogram.reshape(
                sp_shape[0] * sp_shape[3], sp_shape[1], sp_shape[2]
            )

        # FBANK computation
        fbanks = torch.matmul(spectrogram, fbank_matrix)
        if self.log_mel:
            fbanks = self._amplitude_to_DB(fbanks)

        # Reshaping in the case of multi-channel inputs
        if len(sp_shape) == 4:
            fb_shape = fbanks.shape
            fbanks = fbanks.reshape(
                sp_shape[0], fb_shape[1], fb_shape[2], sp_shape[3]
            )

        return fbanks

    @staticmethod
    def _to_mel(hz):
        """Returns mel-frequency value corresponding to the input
        frequency value in Hz.

        Arguments
        ---------
        x : float
            The frequency point in Hz.
        """
        return 2595 * math.log10(1 + hz / 700)

    @staticmethod
    def _to_hz(mel):
        """Returns hz-frequency value corresponding to the input
        mel-frequency value.

        Arguments
        ---------
        x : float
            The frequency point in the mel-scale.
        """
        return 700 * (10 ** (mel / 2595) - 1)

    def _triangular_filters(self, all_freqs, f_central, band):
        """Returns fbank matrix using triangular filters.

        Arguments
        ---------
        all_freqs : Tensor
            Tensor gathering all the frequency points.
        f_central : Tensor
            Tensor gathering central frequencies of each filter.
        band : Tensor
            Tensor gathering the bands of each filter.
        """

        # Computing the slops of the filters
        all_freqs=all_freqs.cuda()
        slope = (all_freqs - f_central) / band
        left_side = slope + 1.0
        right_side = -slope + 1.0

        # Adding zeros for negative values
        zero = torch.zeros(1).cuda()
        # zero = torch.zeros(1, device=self.device_inp)
        fbank_matrix = torch.max(
            zero, torch.min(left_side, right_side)
        ).transpose(0, 1)

        return fbank_matrix

    def _rectangular_filters(self, all_freqs, f_central, band):
        """Returns fbank matrix using rectangular filters.

        Arguments
        ---------
        all_freqs : Tensor
            Tensor gathering all the frequency points.
        f_central : Tensor
            Tensor gathering central frequencies of each filter.
        band : Tensor
            Tensor gathering the bands of each filter.
        """

        # cut-off frequencies of the filters
        low_hz = f_central - band
        high_hz = f_central + band

        # Left/right parts of the filter
        left_side = right_size = all_freqs.ge(low_hz)
        right_size = all_freqs.le(high_hz)

        fbank_matrix = (left_side * right_size).float().transpose(0, 1)

        return fbank_matrix

    def _gaussian_filters(
        self, all_freqs, f_central, band, smooth_factor=torch.tensor(2)
    ):
        """Returns fbank matrix using gaussian filters.

        Arguments
        ---------
        all_freqs : Tensor
            Tensor gathering all the frequency points.
        f_central : Tensor
            Tensor gathering central frequencies of each filter.
        band : Tensor
            Tensor gathering the bands of each filter.
        smooth_factor: Tensor
            Smoothing factor of the gaussian filter. It can be used to employ
            sharper or flatter filters.
        """
        all_freqs=all_freqs.cuda()
        fbank_matrix = torch.exp(
            -0.5 * ((all_freqs - f_central) / (band / smooth_factor)) ** 2
        ).transpose(0, 1)

        return fbank_matrix

    def _create_fbank_matrix(self, f_central_mat, band_mat):
        """Returns fbank matrix to use for averaging the spectrum with
           the set of filter-banks.

        Arguments
        ---------
        f_central : Tensor
            Tensor gathering central frequencies of each filter.
        band : Tensor
            Tensor gathering the bands of each filter.
        smooth_factor: Tensor
            Smoothing factor of the gaussian filter. It can be used to employ
            sharper or flatter filters.
        """
        if self.filter_shape == "triangular":
            fbank_matrix = self._triangular_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )

        elif self.filter_shape == "rectangular":
            fbank_matrix = self._rectangular_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )

        else:
            fbank_matrix = self._gaussian_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )

        return fbank_matrix

    def _amplitude_to_DB(self, x):
        """Converts  linear-FBANKs to log-FBANKs.

        Arguments
        ---------
        x : Tensor
            A batch of linear FBANK tensors.

        """
        x_db = self.multiplier * torch.log10(torch.clamp(x, min=self.amin))
        x_db -= self.multiplier * self.db_multiplier

        # Setting up dB max
        new_x_db_max = x_db.max() - self.top_db
        # Clipping to dB max
        x_db = torch.max(x_db, new_x_db_max)

        return x_db

class FastAudio(torch.nn.Module):
    """Generate features for input to the speech pipeline.

    Arguments
    ---------
    deltas : bool (default: False)
        Whether or not to append derivatives and second derivatives
        to the features.
    context : bool (default: False)
        Whether or not to append forward and backward contexts to
        the features.
    requires_grad : bool (default: False)
        Whether to allow parameters (i.e. fbank centers and
        spreads) to update during training.
    sample_rate : int (default: 160000)
        Sampling rate for the input waveforms.
    f_min : int (default: 0)
        Lowest frequency for the Mel filters.
    f_max : int (default: None)
        Highest frequency for the Mel filters. Note that if f_max is not
        specified it will be set to sample_rate // 2.
    win_length : float (default: 25)
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float (default: 10)
        Length (in ms) of the hop of the sliding window used to compute
        the STFT.
    n_fft : int (default: 400)
        Number of samples to use in each stft.
    n_mels : int (default: 40)
        Number of Mel filters.
    filter_shape : str (default: triangular)
        Shape of the filters ('triangular', 'rectangular', 'gaussian').
    param_change_factor : float (default: 1.0)
        If freeze=False, this parameter affects the speed at which the filter
        parameters (i.e., central_freqs and bands) can be changed.  When high
        (e.g., param_change_factor=1) the filters change a lot during training.
        When low (e.g. param_change_factor=0.1) the filter parameters are more
        stable during training.
    param_rand_factor : float (default: 0.0)
        This parameter can be used to randomly change the filter parameters
        (i.e, central frequencies and bands) during training.  It is thus a
        sort of regularization. param_rand_factor=0 does not affect, while
        param_rand_factor=0.15 allows random variations within +-15% of the
        standard values of the filter parameters (e.g., if the central freq
        is 100 Hz, we can randomly change it from 85 Hz to 115 Hz).
    left_frames : int (default: 5)
        Number of frames of left context to add.
    right_frames : int (default: 5)
        Number of frames of right context to add.

    Example
    -------
    >>> import torch
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = Fbank()
    >>> feats = feature_maker(inputs)
    >>> feats.shape
    torch.Size([10, 101, 40])
    """

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
        sort=False
    ):
        super().__init__()
        self.deltas = deltas
        self.context = context
        self.requires_grad = requires_grad
        self.sort=sort

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
            sort=sort
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
        # with torch.no_grad():

        STFT = self.compute_STFT(wav)
        mag = spectral_magnitude(STFT)
        fbanks = self.compute_fbanks(mag)

        if self.deltas:
            delta1 = self.compute_deltas(fbanks)
            delta2 = self.compute_deltas(delta1)
            fbanks = torch.cat([fbanks, delta1, delta2], dim=2)

        if self.context:
            fbanks = self.context_window(fbanks)

        return fbanks