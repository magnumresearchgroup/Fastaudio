# ASVspoof2021-Fastaudio

FastAudio is team Magnum's solution for the **[ASVspoof 2021 challenge]**. The solution was produced by Quchen Fu and Zhongwei Teng, researchers in the **[Magnum Research Group]** at Vanderbilt University. The Magnum Research Group is part of the **[Institute for Software Integrated Systems]**. 

The ASVspoof 2021 Competition challenges teams to develop countermeasures capable of discriminating between bona fide and spoofed or deepfake speech. The model achieved a 0.2531 min t-DCF score in LA Track on the open **[Leaderboard]**.

[ASVspoof 2021 challenge]: https://www.asvspoof.org
[Magnum Research Group]:https://www.magnum.io
[Institute for Software Integrated Systems]:https://www.isis.vanderbilt.edu
[leaderboard]: https://competitions.codalab.org/competitions/32343#results

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
<!-- 1. Run `python3 main.py --mode preprocess --data_dir src/data --data_file nl2bash-data.json` and `cd src/model && onmt_build_vocab -config nl2cmd.yaml -n_sample 10347 --src_vocab_threshold 2 --tgt_vocab_threshold 2` to process raw data.
2. You can also download the Original raw data [here](https://ibm.ent.box.com/v/nl2bash-data) -->


### Train
1. ``python3.8 train_spoofspeech.py yaml/SpoofSpeechClassifier.yaml --data_parallel_backend --data_parallel_count=2``


### Inference
1. Modify the `TRAIN` in `train_spoofspeech.py` to `False`.
2. ``python3.8 train_spoofspeech.py yaml/SpoofSpeechClassifier.yaml --data_parallel_backend --data_parallel_count=2``

## Reference
If you use this repository, please consider citing:

```
@inproceedings{Fu2021FastAudioAL,
  title={FastAudio: A Learnable Audio Front-End for Spoof Speech Detection},
  author={Quchen Fu and Zhongwei Teng and Jules White and M. Powell and Douglas C. Schmidt},
  year={2021}
}
```