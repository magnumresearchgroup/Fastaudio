# ASVspoof2021-Fastaudio#


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
2. Modify the `world_size` in `src/model/nl2cmd.yaml` to the number of GPUs you are using and put the ids as `gpu_ranks`.


### Inference
1. Modify the `TRAIN` in `train_spoofspeech.py` to `False`.

2. `onmt_translate -model src/model/run/model_step_2000.pt -src src/data/invocations_proccess_test.txt -output pred_2000.txt -gpu 0 -verbose`
