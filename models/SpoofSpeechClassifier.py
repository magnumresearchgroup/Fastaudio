#!/usr/bin/env python3
import torch
import speechbrain as sb
from torch.utils.data import DataLoader
from speechbrain import Stage
from tqdm.contrib import tqdm
import os
import pandas as pd
import wandb
import numpy as np
import json
import losses.eval_metrics as em


class SpoofSpeechClassifier(sb.Brain):
    def compute_forward(self, batch, stage):
        """Runs all the computation of that transforms the input into the
        output probabilities over the N classes.
        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        predictions : Tensor
            Tensor that contains the posterior probabilities over the N classes.
        """

        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)
        # Compute features, embeddings, and predictions
        feats, lens = self.prepare_features(batch.sig, stage)
        embeddings = self.modules.embedding_model(feats.cuda(), lens.cuda())
        predictions = self.modules.classifier(embeddings)
        return predictions

    def prepare_features(self, wavs, stage):
        """Prepare the features for computation, including augmentation.
        Arguments
        ---------
        wavs : tuple
            Input signals (tensor) and their relative lengths (tensor).
        stage : sb.Stage
            The current stage of training.
        """
        wavs, lens = wavs
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                lens = torch.cat([lens, lens])

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, lens)

        # Choose what features we want to use
        # todo: support multiple features and feature concat
        target_feats = self.hparams.embedding_features

        FEATURE_EXTRACTOR = {
            # 'cqt': self.modules.cqt,
            # 'fbanks': self.modules.fbanks
            'fastaudiogauss': self.modules.fastaudiogauss
            # 'ifr': self.modules.ifr
            # 'mag': self.modules.mag
            # 'mfcc': self.modules.mfcc
            # 'leaf': self.modules.leaf
            # 'tdfbanks': self.modules.tdfbanks
            # 'pcen': self.modules.pcen
            # 'sincnet': self.modules.sincnet
            # 'trainable_fbanks': self.modules.trainable_fbanks
        }

        if len(target_feats) == 1:
            # wavs = wavs.unsqueeze(1).cuda()
            feats = FEATURE_EXTRACTOR[target_feats[0]](wavs)
            # feats = torch.unsqueeze(feats, 1)
            # feats = torch.transpose(feats, 1,2)
            if target_feats[0]=='cqt':
                log_spec = 10.0 * torch.log10(torch.clamp(feats, min=1e-30))
                log_spec -= 10.0
                feats=log_spec
                feats = torch.transpose(feats, 1,2)
        else:
            feats = []
            for target in target_feats:
                temp = FEATURE_EXTRACTOR[target](wavs)
                if target=='cqt':
                    temp = torch.transpose(temp, 1,2)
                feats.append(temp)
            f =feats[0]
            for i in range(1, len(feats)):
                f = torch.cat((f, feats[i]), dim=2)
            feats = f
        feats = self.modules.mean_var_norm(feats, lens)
        return feats, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        Arguments
        ---------
        predictions : tensor
            The output tensor from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        _, lens = batch.sig
        spkid, _ = batch.key_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hasattr(self.modules, "env_corrupt"):
            spkid = torch.cat([spkid, spkid], dim=0)
            lens = torch.cat([lens, lens])
        # Compute the cost function
        loss = sb.nnet.losses.bce_loss(predictions, spkid, lens)

        # Compute classification error at test time
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, spkid, lens)
        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()
        if stage == sb.Stage.VALID:
            label_encoder = sb.dataio.encoder.CategoricalEncoder()

            lab_enc_file = os.path.join(self.hparams.save_folder, "label_encoder.txt")

            label_encoder.load(path=lab_enc_file)

            self.bona_index = label_encoder.encode_label('bonafide')
            self.spoof_index = label_encoder.encode_label('spoof')

            self.pd_out = {'files': [], 'scores': []}

    def on_stage_end(self, stage, stage_loss, epoch=None):
        def compute_det_curve(target_scores, nontarget_scores):

            n_scores = target_scores.size + nontarget_scores.size
            all_scores = np.concatenate((target_scores, nontarget_scores))
            labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

            # Sort labels based on scores
            indices = np.argsort(all_scores, kind='mergesort')
            labels = labels[indices]

            # Compute false rejection and false acceptance rates
            tar_trial_sums = np.cumsum(labels)
            nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

            frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
            far = np.concatenate(
                (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
            thresholds = np.concatenate(
                (
                np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

            return frr, far, thresholds

        def compute_eer(target_scores, nontarget_scores):
            """ Returns equal error rate (EER) and the corresponding threshold. """
            frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
            abs_diffs = np.abs(frr - far)
            min_index = np.argmin(abs_diffs)
            eer = np.mean((frr[min_index], far[min_index]))
            return eer, thresholds[min_index]

        def get_eer_tDCF(asv_score_file='losses/LA.asv.dev.scores.txt',
                     cm_target_score_file='predictions/target_score.json',
                     cm_nontarget_score_file='predictions/nontarget_score.json'
                     ):
            Pspoof = 0.05
            cost_model = {
                'Pspoof': Pspoof,  # Prior probability of a spoofing attack
                'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
                'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
                'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
                'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
                'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
                'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
            }
            # Load organizers' ASV scores
            asv_data = np.genfromtxt(asv_score_file, dtype=str)
            asv_sources = asv_data[:, 0]
            asv_keys = asv_data[:, 1]
            asv_scores = asv_data[:, 2].astype(np.float)

            # Extract target, nontarget, and spoof scores from the ASV scores
            tar_asv = asv_scores[asv_keys == 'target']
            non_asv = asv_scores[asv_keys == 'nontarget']
            spoof_asv = asv_scores[asv_keys == 'spoof']

            # Extract bona fide (real human) and spoof scores from the CM scores

            with open(cm_target_score_file, 'r') as f:
                bona_cm = np.array(json.load(f)['score'])
            with open(cm_nontarget_score_file, 'r') as f:
                spoof_cm = np.array(json.load(f)['score'])

            # EERs of the standalone systems and fix ASV operating point to EER threshold
            eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)
            eer_cm = compute_eer(bona_cm, spoof_cm)[0]

            [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv,
                                                                              asv_threshold)

            # Compute t-DCF
            tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv,
                                                        cost_model,
                                                        False)

            # Minimum t-DCF
            min_tDCF_index = np.argmin(tDCF_curve)
            min_tDCF = tDCF_curve[min_tDCF_index]
            return eer_cm, min_tDCF


        def split_target_non_target():
            pred_file = 'predictions/scores.txt'
            gt_file = 'processed_data/la_cm_dev.json'
            with open(gt_file, 'r') as f:
                gt = json.load(f)

            with open(pred_file, 'r') as f:
                preds = f.readlines()

            target_scores = []
            non_target_scores = []

            for pred in preds:
                i, score = pred.split()
                score = float(score)

                if gt[i]['key'] == 'spoof':
                    non_target_scores.append(score)
                else:
                    target_scores.append(score)
            with open('predictions/target_score.json', 'w') as f:
                json.dump({'score': target_scores}, f)
            with open('predictions/nontarget_score.json', 'w') as f:
                json.dump({'score': non_target_scores}, f)



        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            wandb.log({"train_loss": self.train_loss})

        # At the end of validation...
        if stage == sb.Stage.VALID:

            # old_lr, new_lr = self.hparams.lr_annealing(epoch)
            old_lr, new_lr = self.hparams.lr_scheduler([self.optimizer], epoch, stage_loss)
            # new_lr=self.hparams.lr_annealing(self.optimizer)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            # self.hparams.lr_annealing()
            # The train_logger writes a summary to stdout and to the logfile.
            pd.DataFrame(self.pd_out).to_csv('predictions/scores.txt', sep=' ', header=False, index=False)
            split_target_non_target()
            eer_cm, min_tDCF = get_eer_tDCF()

            stats = {
                "loss": stage_loss,
                "min_tDCF": min_tDCF,
            }
            wandb.log({"stage_loss": stage_loss})
            wandb.log({"error": self.error_metrics.summarize("average")})
            # Save the current checkpoint and delete previous checkpoints,

            wandb.log({"eer_cm": eer_cm})
            wandb.log({"min_tDCF": min_tDCF})

            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": new_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # self.checkpointer.save_and_keep_only(meta=stats, min_keys=["min_tDCF"])
            self.checkpointer.save_and_keep_only(meta=stats, num_to_keep=5, keep_recent=True)

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            stats = {
                "loss": stage_loss,
                "error": self.error_metrics.summarize("average"),
            }
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


    def evaluate_batch(self, batch, stage):
        """
        Overwrite evaluate_batch.
        Keep same for stage in (TRAIN, VALID)
        Output probability in TEST stage (from classify_batch)
        """
        if stage != sb.Stage.TEST:
            # Same as before
            out = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(out, batch, stage=stage)
            out_prob = self.compute_forward(batch, stage=stage)
            out_prob = out_prob.squeeze(1)
            score, index = torch.max(out_prob, dim=-1)
            cm_scores = [out_prob[i].item() for i in range(out_prob.shape[0])]
            self.pd_out['files'] += batch.id
            self.pd_out['scores'] += cm_scores
            return loss.detach().cpu()
        else:
            out_prob = self.compute_forward(batch, stage=stage)
            out_prob = out_prob.squeeze(1)
            score, index = torch.max(out_prob, dim=-1)
            # text_lab = self.hparams.label_encoder.decode_torch(index)
            return out_prob, score, index
            # return out_prob, score, index, text_lab

    def evaluate(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
        run_opts={"device": "cuda"}
    ):
        def compute_det_curve(target_scores, nontarget_scores):

            n_scores = target_scores.size + nontarget_scores.size
            all_scores = np.concatenate((target_scores, nontarget_scores))
            labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

            # Sort labels based on scores
            indices = np.argsort(all_scores, kind='mergesort')
            labels = labels[indices]

            # Compute false rejection and false acceptance rates
            tar_trial_sums = np.cumsum(labels)
            nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

            frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
            far = np.concatenate(
                (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
            thresholds = np.concatenate(
                (
                np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

            return frr, far, thresholds

        def compute_eer(target_scores, nontarget_scores):
            """ Returns equal error rate (EER) and the corresponding threshold. """
            frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
            abs_diffs = np.abs(frr - far)
            min_index = np.argmin(abs_diffs)
            eer = np.mean((frr[min_index], far[min_index]))
            return eer, thresholds[min_index]

        def get_eer_tDCF(asv_score_file='losses/LA.asv.dev.scores.txt',
                     cm_target_score_file='predictions/nontarget_score.json',
                     cm_nontarget_score_file='predictions/nontarget_score.json'
                     ):
            Pspoof = 0.05
            cost_model = {
                'Pspoof': Pspoof,  # Prior probability of a spoofing attack
                'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
                'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
                'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
                'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
                'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
                'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
            }
            # Load organizers' ASV scores
            asv_data = np.genfromtxt(asv_score_file, dtype=str)
            asv_sources = asv_data[:, 0]
            asv_keys = asv_data[:, 1]
            asv_scores = asv_data[:, 2].astype(np.float)

            # Extract target, nontarget, and spoof scores from the ASV scores
            tar_asv = asv_scores[asv_keys == 'target']
            non_asv = asv_scores[asv_keys == 'nontarget']
            spoof_asv = asv_scores[asv_keys == 'spoof']

            # Extract bona fide (real human) and spoof scores from the CM scores

            with open(cm_target_score_file, 'r') as f:
                bona_cm = np.array(json.load(f)['score'])
            with open(cm_nontarget_score_file, 'r') as f:
                spoof_cm = np.array(json.load(f)['score'])

            # EERs of the standalone systems and fix ASV operating point to EER threshold
            eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)
            eer_cm = compute_eer(bona_cm, spoof_cm)[0]

            [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv,
                                                                              asv_threshold)

            # Compute t-DCF
            tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv,
                                                        cost_model,
                                                        False)

            # Minimum t-DCF
            min_tDCF_index = np.argmin(tDCF_curve)
            min_tDCF = tDCF_curve[min_tDCF_index]
            return eer_cm, min_tDCF


        def split_target_non_target():
            pred_file = 'predictions/scores.txt'
            gt_file = 'processed_data/la_cm_dev.json'
            with open(gt_file, 'r') as f:
                gt = json.load(f)

            with open(pred_file, 'r') as f:
                preds = f.readlines()

            target_scores = []
            non_target_scores = []

            for pred in preds:
                i, score = pred.split()
                score = float(score)

                if gt[i]['key'] == 'spoof':
                    non_target_scores.append(score)
                else:
                    target_scores.append(score)
            with open('predictions/target_score.json', 'w') as f:
                json.dump({'score': target_scores}, f)
            with open('predictions/nontarget_score.json', 'w') as f:
                json.dump({'score': non_target_scores}, f)

        if progressbar is None:
            progressbar = not self.noprogressbar

        if not isinstance(test_set, DataLoader):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0

        label_encoder = sb.dataio.encoder.CategoricalEncoder()

        lab_enc_file = os.path.join(self.hparams.save_folder, "label_encoder.txt")

        label_encoder.load(path=lab_enc_file)

        bona_index = label_encoder.encode_label('bonafide')
        spoof_index = label_encoder.encode_label('spoof')

        pd_out = {'files': [], 'scores': []}


        with torch.no_grad():
            for batch in tqdm(
                    test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1
                """
                Rewrite here
                bonafide --> 0 , spoof -->
                """
                out_prob, score, index = self.evaluate_batch(batch, stage=Stage.TEST)
                cm_scores = [out_prob[i].item() for i in range(out_prob.shape[0])]
                pd_out['files'] += batch.id
                pd_out['scores'] += cm_scores

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break
                """
                Rewrite Over
                """
        pd.DataFrame(pd_out).to_csv('predictions/scores.txt', sep=' ', header=False, index=False)
        self.step = 0
