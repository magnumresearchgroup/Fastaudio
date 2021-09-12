import numpy as np
import json
import eval_metrics as em
from speechbrain.utils.metric_stats import EER
import torch


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

def compute_tDCF(asv_score_file = 'LA.asv.eval.scores.txt',
                cm_target_score_file = '/home/zhongwei/work2021/google-voice/asv2021-speechbrain/predictions/TDNN_eval_target_score.json',
                 cm_nontarget_score_file = '/home/zhongwei/work2021/google-voice/asv2021-speechbrain/predictions/TDNN_eval_nontarget_score.json'
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

def compute_asv_eer():
    target_score_file = '/home/zhongwei/work2021/google-voice/asv2021-speechbrain/predictions/TDNN_eval_target_score.json'
    nontarget_score_file = '/home/zhongwei/work2021/google-voice/asv2021-speechbrain/predictions/TDNN_eval_nontarget_score.json'
    with open(target_score_file, 'r') as f:
        target_scores = json.load(f)['score']

    with open(nontarget_score_file, 'r') as f:
        nontarget_scores = json.load(f)['score']

    eer, th = compute_eer(np.array(target_scores), np.array(nontarget_scores))
    print(eer)
    print(th)


compute_tDCF()