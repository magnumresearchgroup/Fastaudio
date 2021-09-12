import numpy as np
import json
import losses.eval_metrics as em
import random
from random import randint
from collections import defaultdict

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
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds

def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def get_tDCF(asv_score_file = 'losses/LA.asv.eval.scores.txt',
                cm_target_score_file = 'predictions/TDNN_eval_target_score.json',
                 cm_nontarget_score_file = 'predictions/TDNN_eval_nontarget_score.json'
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

    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

    # Compute t-DCF
    tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model,
                                                False)

    # Minimum t-DCF
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]
    return min_tDCF

def compute_tDCF(asv_score_file = 'losses/LA.asv.eval.scores.txt',
                cm_target_score_file = 'predictions/TDNN_eval_target_score.json',
                 cm_nontarget_score_file = 'predictions/TDNN_eval_nontarget_score.json'
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

    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

    # Compute t-DCF
    print(Pfa_asv)
    print(Pmiss_asv)
    print(Pmiss_spoof_asv)
    tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model,
                                                True)

    # Minimum t-DCF
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    print('ASV SYSTEM')
    print('   EER            = {:8.5f} % (Equal error rate (target vs. nontarget discrimination)'.format(eer_asv * 100))
    print('   Pfa            = {:8.5f} % (False acceptance rate of nontargets)'.format(Pfa_asv * 100))
    print('   Pmiss          = {:8.5f} % (False rejection rate of targets)'.format(Pmiss_asv * 100))
    print('   1-Pmiss,spoof  = {:8.5f} % (Spoof false acceptance rate)'.format((1 - Pmiss_spoof_asv) * 100))

    print('\nCM SYSTEM')
    print('   EER            = {:8.5f} % (Equal error rate for countermeasure)'.format(eer_cm * 100))

    print('\nTANDEM')
    print('   min-tDCF       = {:8.5f}'.format(min_tDCF))

def split_target_non_target():
    pred_file = 'predictions/scores.txt'
    gt_file = 'processed_data/la_cm_eval.json'
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
    with open('predictions/nontarget_score.json','w') as f:
        json.dump({'score': non_target_scores}, f)

def find_best_ratio(scores_list):
    gt_file = 'processed_data/la_cm_eval.json'
    n = len(scores_list)
    with open(gt_file, 'r') as f:
        gt = json.load(f)
    min_ratio = None
    min_t  = 1000

    for _ in range(500):
        ratio = [randint(1, 100) for _ in range(n)]
        s = sum(ratio)
        ratio = [r/float(s) for r in ratio]

        assert len(ratio) == len(scores_list)

        scores = defaultdict(int)

        for i in range(n):
            with open(scores_list[i], 'r') as f:
                preds = f.readlines()
                for pred in preds:
                    audio_id, score = pred.split()
                    score = float(score)
                    scores[audio_id] += float(score) * ratio[i]

        target_scores = []
        non_target_scores = []
        for audio_id in scores:
            if gt[audio_id]['key'] == 'spoof':
                non_target_scores.append(scores[audio_id])
            else:
                target_scores.append(scores[audio_id])
        with open('predictions/target_score.json', 'w') as f:
            json.dump({'score': target_scores}, f)
        with open('predictions/nontarget_score.json','w') as f:
            json.dump({'score': non_target_scores}, f)
        cur_t = get_tDCF(asv_score_file='losses/LA.asv.eval.scores.txt',
                             cm_target_score_file='predictions/target_score.json',
                             cm_nontarget_score_file='predictions/nontarget_score.json')
        if cur_t<min_t:
            min_ratio = ratio
            min_t=cur_t
    with open('predictions/min_ratio.txt','w') as f:
        f.write(str(min_ratio))

    print(min_ratio)
    print(min_t)
    return min_ratio

split_target_non_target()
compute_tDCF(asv_score_file = 'losses/LA.asv.eval.scores.txt',
             cm_target_score_file = 'predictions/target_score.json',
             cm_nontarget_score_file = 'predictions/nontarget_score.json')