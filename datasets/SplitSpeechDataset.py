"""
This file is a modification of original SpeechSplit Model
With SpeechBrain DynamicDataset
"""


import os
import torch
import pickle
import numpy as np
import json
import copy

from functools import partial
from numpy.random import uniform
from multiprocessing import Process, Manager

from torch.utils import data
from torch.utils.data.sampler import Sampler

import speechbrain as sb
from pysptk import sptk #Speech Signal Processing Toolkit (SPTK)\
from pathlib import Path
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch

def get_split_speech_loader(hparams):
    # Initialization of the label encoder.
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define fbank  pipeline
    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("fbank")
    def fbank_pipeline(file_path):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`.
         Audio tensor with shape: (samples, )
        """
        path = Path(file_path)
        file_id = path.stem
        root = str(path.parent.parent.absolute())
        root= Path(root.replace('data','extract_data'))
        fbank_path = os.path.join(root, 'fbank', file_id+'.npy')
        fbank = torch.tensor(np.load(fbank_path))
        # It should be flatten
        fbank = fbank.transpose(0, 1)
        fbank = fbank.squeeze(1)
        return fbank.squeeze(1)
        # return torch.tensor(fbank)


    # Define audio pipeline
    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(file_path):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(file_path)
        return sig


    # Define f0_rapt pipelien: fundamental frequency_robust algorithm for pitch tracking
    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides('f0')
    def f0_pipeline(file_path):
        """
        Load the signal and use pysptk to extract f0/pitch information
        """
        path = Path(file_path)
        file_id = path.stem
        root = str(path.parent.parent.absolute())
        root = Path(root.replace('data', 'extract_data'))
        f0_path = os.path.join(root, 'f0', file_id + '.npy')
        f0 = torch.tensor(np.load(f0_path))
        return f0
        # It should be flatten
        # f0 = f0.transpose(0, 1)
        # return f0.squeeze(1)

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("key")
    @sb.utils.data_pipeline.provides("key", "key_encoded")
    def label_pipeline(key):
        yield key
        key_encoded = label_encoder.encode_label_torch(key)
        yield key_encoded

    datasets = {}
    for dataset in ["train", "dev", "eval"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, f0_pipeline,label_pipeline],
            output_keys=["id", "sig", "f0", "key_encoded", "key"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="key",
    )

    dataloaders = {}

    for k in datasets:
        dataset = datasets[k]
        dataloaders[k] = SaveableDataLoader(dataset, batch_size=hparams['batch_size'],
                                            collate_fn=PaddedBatch)
    return dataloaders

def get_processed_dataloader(hparams):
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define fbank  pipeline
    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("fbank")
    def fbank_pipeline(file_path):
        path = Path(file_path)
        file_id = path.stem
        root = str(path.parent.parent.parent.absolute())
        root= Path(root.replace('data','extract_data'))
        if 'dev' in file_path:
            fbank_path = os.path.join(root, 'dev', 'fbank', file_id+'.npy')
        else:
            fbank_path = os.path.join(root, 'train', 'fbank', file_id + '.npy')
        fbank = torch.tensor(np.load(fbank_path))
        # It should be flatten
        # fbank = fbank.transpose(0, 1)
        fbank = fbank.squeeze(0)
        # return fbank.squeeze(1)
        # return torch.tensor(fbank)
        return fbank

    # Define f0  pipeline
    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("f0")
    def f0_pipeline(file_path):
        path = Path(file_path)
        file_id = path.stem
        root = str(path.parent.parent.parent.absolute())
        root = Path(root.replace('data', 'extract_data'))
        if 'dev' in file_path:
            f0_path = os.path.join(root, 'dev', 'f0', file_id+'.npy')
        else:
            f0_path = os.path.join(root, 'train', 'f0', file_id + '.npy')

        # f0_path = os.path.join(root, 'f0', file_id + '.npy')
        f0 = torch.tensor(np.load(f0_path))
        # It should be flatten
        # fbank = fbank.transpose(0, 1)
        f0 = f0.squeeze(0)
        # return fbank.squeeze(1)
        return f0

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("key")
    @sb.utils.data_pipeline.provides("key", "key_encoded")
    def label_pipeline(key):
        yield key
        key_encoded = label_encoder.encode_label_torch(key)
        yield key_encoded

    datasets = {}
    for dataset in ["train", "dev"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[fbank_pipeline, f0_pipeline,label_pipeline],
            output_keys=["id", "fbank", "f0", "key_encoded", "key"],
        )

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="key",
    )

    dataloaders = {}

    for k in datasets:
        dataset = datasets[k]
        dataloaders[k] = SaveableDataLoader(dataset, batch_size=hparams['batch_size'],
                                            collate_fn=PaddedBatch)
    return dataloaders


def double_corrupt_data_json():
    for p, t in zip(('processed_data/new_train.json','processed_data/new_dev.json'),
                    ('processed_data/train_corrupt.json','processed_data/dev_corrupt.json')
                    ):
        with open(p, 'r') as f:
            data = json.load(f)
            corrupt_data = {}
            for k in data:
                for e in ['_aug', '_ori']:
                    new_k = k + e
                    corrupt_data[new_k] = copy.deepcopy(data[k])
                    corrupt_data[new_k]['file_path'] = corrupt_data[new_k]['file_path'].replace(k, new_k)
        with open(t, 'w') as f:
            json.dump(corrupt_data, f)


