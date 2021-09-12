import os
import glob
from pathlib import Path
from scipy import signal
import speechbrain as sb
from speechbrain.lobes.features import Fbank
import numpy as np
from pysptk import sptk
from numpy.random import RandomState
import soundfile as sf
import torchaudio
from librosa.filters import mel
from utils.signal_process import butter_highpass,speaker_normalization,pySTFT

from speechbrain.utils.data_utils import pad_right_to
import pandas as pd
from speechbrain.lobes.augment import EnvCorrupt
from speechbrain.lobes.augment import TimeDomainSpecAugment
from speechbrain.processing.features import InputNormalization
import torch


def extract_and_save_mel_f0(data_dir = '../data/LA', out_dir = '../extract_data/LA/mel'):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    prng = RandomState(2411)

    # Set up augmentation models
    corrupter = EnvCorrupt(openrir_folder= './processed_data',
                            babble_prob=0.0,
                            reverb_prob=0.0,
                            noise_prob=1.0,
                            noise_snr_low=0,
                            noise_snr_high=15)

    augmentor = TimeDomainSpecAugment(sample_rate=16000,
                                      speeds= [95, 100, 105])

    # Actual hop_length here is  round((self.sample_rate / 1000.0) * self.hop_length)
    fbank_feature_model = Fbank(n_mels=80, f_min=90, f_max=7600, sample_rate=16000,
                                hop_length=16)

    f0_extractor = lambda wavs: torch.Tensor(np.apply_along_axis(lambda x: sptk.rapt(x.astype(np.float32) * 32768,
                                                                                     16000,
                                                                                     256,
                                                                                     min=100,
                                                                                     max=600,
                                                                                     otype=2),
                                                                 1,
                                                                 wavs.numpy()))

    norm = InputNormalization(norm_type='sentence',std_norm=False)


    # Find maximum audio length, which is the pad length
    for split in ['train']:
        # root directory of ASV data
        root_dir = os.path.join(data_dir, 'ASVspoof2019_LA_'+split, 'flac', '*.flac')
        # create outpur directory
        Path(os.path.join(out_dir,split)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(out_dir, split,'fbank')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(out_dir, split, 'f0')).mkdir(parents=True, exist_ok=True)

        # audio file path list
        flac_files = glob.glob(root_dir, recursive=True)
        max_pad_len = 0
        # distribution of audio length
        # Check if max pad length has been saved:
        if Path(os.path.join(out_dir, 'audio_len.csv')).is_file():
            max_pad_len = max(pd.read_csv(os.path.join(out_dir, 'audio_len.csv'))['audio_len'])
        else:
            audio_len_dist = []
            for file in flac_files:
                sig, sample_rate = torchaudio.load(file)
                max_pad_len = max(sig.shape[1], max_pad_len)
                audio_len_dist.append(sig.shape[1])
            # Write length distribution into a csv file
            pd.DataFrame({'audio_len': audio_len_dist}).to_csv(
                os.path.join(out_dir, 'audio_len.csv'),
                index=False)

        print('Max pad length is: %s'%(max_pad_len))


        f_shape = None



        def bad_pad(t, shape):
            if t.shape[1] > shape[1]:
                t = t.narrow(1, 0, shape[1]+1)
            else:
                t, _ = pad_right_to(t, shape)
            return t


        # Start to Padding
        pad_shape = (1, max_pad_len)
        for file in flac_files:
            sig, sample_rate = torchaudio.load(file)
            padded_sig, val_sig = pad_right_to(sig, pad_shape)
            # convert padded signal into Fbanks and F0 with different modules
            # Data augmentation
            #todo: torch.ones(1) is shapes, need to check here
            lens = torch.ones(1)
            aug_sig = corrupter(padded_sig, lens)

            con_sig = torch.cat([aug_sig, padded_sig], dim=0)
            con_sig = augmentor(con_sig, torch.cat([lens, lens]))



            aug_sig = con_sig[0].unsqueeze(0)
            ori_sig = con_sig[1].unsqueeze(0)

            aug_sig = bad_pad(aug_sig, pad_shape)
            ori_sig = bad_pad(ori_sig, pad_shape)

            fbank_aug = norm(fbank_feature_model(aug_sig), lens).numpy()
            f0_aug = norm(f0_extractor(aug_sig),lens).numpy()

            fbank_ori = norm(fbank_feature_model(ori_sig), lens).numpy()
            f0_ori = norm(f0_extractor(ori_sig), lens).numpy()

            if f_shape == None:
                f_shape = fbank_aug.shape[1]


            assert fbank_aug.shape[1] == f_shape
            assert fbank_ori.shape[1] == f_shape
            assert f0_aug.shape[1] == f_shape
            assert f0_aug.shape[1] == f_shape


            id = Path(file).stem
            f0_out_file = os.path.join(out_dir, split,'f0', id)
            fbank_out_file = os.path.join(out_dir, split, 'fbank', id)

            np.save(fbank_out_file+'_aug',
                    fbank_aug.astype(np.float32),
                    allow_pickle=False)
            np.save(f0_out_file+'_aug',
                    f0_aug.astype(np.float32),
                    allow_pickle=False)

            np.save(fbank_out_file+'_ori',
                    fbank_ori.astype(np.float32), allow_pickle=False)
            np.save(f0_out_file+'_ori',
                    f0_ori.astype(np.float32),
                    allow_pickle=False)





            pass


        # for file in flac_files:
        #     id = Path(file).stem
        #     f0_out_file = os.path.join(out_dir, split,'f0', id)
        #     fbank_out_file = os.path.join(out_dir, split, 'fbank', id)
        #
        #     fbank = extract_mel(file, fbank_feature_model)[0]
        #     f0 = extract_f0(file, prng)
        #     assert len(fbank) == len(f0)
        #     np.save(fbank_out_file, fbank.astype(np.float32), allow_pickle=False)
        #     np.save(f0_out_file, f0.astype(np.float32), allow_pickle=False)
        #     # break



        #todo: is it okay we don't put corrupt data and original data in the same batch?










def extract_mel(file, fbank_feature_model):
    audio, fs = torchaudio.load(file)
    fbank = fbank_feature_model(audio).numpy()
    return fbank

def extract_f0(file, prng):

    x, fs = sf.read(file)
    b, a = butter_highpass(30, 16000, order=5)
    if x.shape[0] % 256 == 0:
        x = np.concatenate((x, np.array([1e-06])), axis=0)
    y = signal.filtfilt(b, a, x)
    wav = y * 0.96 + (prng.rand(y.shape[0]) - 0.5) * 1e-06


    f0_rapt = sptk.rapt(wav.astype(np.float32) * 32768, fs, 256, min=100, max=600, otype=2)
    index_nonzero = (f0_rapt != -1e10)
    mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
    f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)

    return f0_norm



# extract_and_save_mel_f0()


