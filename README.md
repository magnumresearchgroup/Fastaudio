# Fastaudio

FastAudio is a **[Learnable Audio Frontend]** team Magnum's designed for the **[ASVspoof 2021 challenge]**. It was developed using the **[Speechbrain]** framework. The solution was produced by Quchen Fu and Zhongwei Teng, researchers in the **[Magnum Research Group]** at Vanderbilt University. The Magnum Research Group is part of the **[Institute for Software Integrated Systems]**. 

The ASVspoof 2021 Competition challenges teams to develop countermeasures capable of discriminating between bona fide and spoofed or deepfake speech. The model achieved a 0.2531 min t-DCF score in LA Track on the open **[Leaderboard]**.

[Learnable Audio Frontend]: https://arxiv.org/abs/2109.02774
[ASVspoof 2021 challenge]: https://www.asvspoof.org
[Magnum Research Group]:https://www.magnum.io
[Institute for Software Integrated Systems]:https://www.isis.vanderbilt.edu
[leaderboard]: https://competitions.codalab.org/competitions/32343#results
[Speechbrain]: https://github.com/speechbrain/speechbrain.git

## Requirements
<details><summary>Show details</summary>
<p>

* speechbrain==0.5.7
* pandas
* wandb
* torch==1.8.0+cu111
* torchaudio==0.8.0

</p>
</details>

## How it works

### Environment
1. Create a virtual environment with python3.8 installed(`virtualenv`)
2. ``git clone --recursive https://github.com/QuchenFu/Fastaudio``
3. use `pip install -r requirements.txt` to install the requirements files.
4. ``cd leaf-audio-pytorch/`` and ``pip install -e .``
5. ``pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html``

### Data pre-processing

    .
    ├── data                       
    │   │
    │   ├── PA                  
    │   │   └── ...
    │   └── LA           
    │       ├── ASVspoof2019_LA_asv_protocols
    │       ├── ASVspoof2019_LA_asv_scores
    │       ├── ASVspoof2019_LA_cm_protocols
    │       ├── ASVspoof2019_LA_train
    │       ├── ASVspoof2019_LA_dev
    │       └── ASVspoof2021_LA_eval
    │
    └── Fastaudio

1. Download the data [here](https://zenodo.org/record/4837263)
2. Unzip and save the data to a folder `data` in the same directory as `Fastaudio`
3. ``python3.8 preprocess.py``
4. Change ``args['data_type'] = ['labeled','unlabeled'][1]`` in ``preprocess.py`` to ``args['data_type'] = ['labeled','unlabeled'][0]``
5. ``python3.8 preprocess.py``

### Train
1. ``python3.8 train_spoofspeech.py yaml/SpoofSpeechClassifier.yaml --data_parallel_backend --data_parallel_count=2``


### Inference
1. Modify the `TRAIN` in `train_spoofspeech.py` to `False`.
2. ``python3.8 train_spoofspeech.py yaml/SpoofSpeechClassifier.yaml --data_parallel_backend --data_parallel_count=2``


### Evaluate
1. ``python3.8 eval.py``

## Reference
If you use this repository, please consider citing:

```
@inproceedings{Fu2021FastAudioAL,
  title={FastAudio: A Learnable Audio Front-End for Spoof Speech Detection},
  author={Quchen Fu and Zhongwei Teng and Jules White and M. Powell and Douglas C. Schmidt},
  year={2021}
}
```

```
@inproceedings{Teng2021ComplementingHF,
  title={Complementing Handcrafted Features with Raw Waveform Using a Light-weight Auxiliary Model},
  author={Zhongwei Teng and Quchen Fu and Jules White and M. Powell and Douglas C. Schmidt},
  year={2021}
}
```