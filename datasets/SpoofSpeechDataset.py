import speechbrain as sb
import os

def get_dataset(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `prepare_mini_librispeech` to have been called before this,
    so that the `train.json`, `valid.json`,  and `valid.json` manifest files
    are available.
    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.
    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(file_path):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""

        sig = sb.dataio.dataio.read_audio(file_path)
        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("key")
    @sb.utils.data_pipeline.provides("key", "key_encoded")
    def label_pipeline(key):
        yield key
        key_encoded = label_encoder.encode_label_torch(key)
        yield key_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    hparams["dataloader_options"]["shuffle"] = True

    for dataset in ["train", "dev", "eval"]:
        print(hparams[f"{dataset}_annotation"])
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "key_encoded", "key"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        # from_didatasets=[datasets["train"]],
        from_iterables=[('spoof','bonafide')],
        output_key="key",
    )
    return datasets