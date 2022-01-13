"""

Given compiled predictions (see the `collect_organize_preds_across_folds.py` output),
compare them with the original dataset gold contexts, and produce metrics.


We output 2 types of metrics at 2 types of aggregation (across papers).

    - Metrics are either sentence-level (weak) or context-level (strict)
    - Aggregations are either macro or micro-averaged across papers.


Note difference in JSON formats:
    - Predictions are at a sentence level.  We don't do any experiments w/ predicting entire contexts.
    - Gold is at a context level (which can be unfurled to be at a sentence level).


Run script:

    gold_data="data/full-v20210918.json"
    pred_dir="oracle_test_val_ckpts_epochs_0_1_2_3"
    target_dir=${pred_dir}
    intent="__nointent"
    for window in 1 3 5 7 9 11
    do
        data="allenai-scibert_scivocab_uncased__${window}__1__07-01-02"
        model="${data}__batch32${intent}"
        for ckpt in 0 1 2 3
        do
            python seq_tagger/eval_seq_tagger_preds.py \
            --pred ${pred_dir}/${model}/all_preds_for_test-${ckpt}.json \
            --gold ${gold_data} \
            --output ${target_dir}/${model}/results_for_test-${ckpt}.json
        done
    done

*NOTE* adapt this for the __nointent__ cases as well


"""

from typing import Dict, Tuple, List, Set

import json
import argparse
from collections import defaultdict

import numpy as np

try:
    from seq_tagger.const import INTENT_TOKENS
except ImportError:
    from const import INTENT_TOKENS



def _tp_fp_fn_given_paper(pred_dict_for_paper: Dict, gold_dict_for_paper: Dict) -> Tuple[int, int, int]:
    tp, fp, fn = 0, 0, 0

    # convert to gold collection for sent-level evaluation
    gold_intent_to_sents: Dict[str, Set[str]] = defaultdict(set)
    for intent, d in gold_dict_for_paper['y'].items():
        for context in d['gold_contexts']:
            for sent_id in context:
                gold_intent_to_sents[intent].add(sent_id)

    # count predictions that match gold
    for intent, sent_ids in pred_dict_for_paper.items():
        for sent_id in sent_ids:
            if intent in gold_intent_to_sents and sent_id in gold_intent_to_sents[intent]:
                tp += 1
            else:
                fp += 1

    # count gold that we missed
    for intent, sent_ids in gold_intent_to_sents.items():
        for sent_id in sent_ids:
            if intent not in pred_dict_for_paper:
                fn += 1
            elif sent_id not in pred_dict_for_paper[intent]:
                fn += 1
    return tp, fp, fn


def _test_accum():
    pred_dict_for_paper = {'@BACK@': ['1', '2', '3'], '@USE@': ['4', '5']}

    # exact
    gold_dict_for_paper = {'x': [], 'y': {'@BACK@': {'gold_contexts': [['1', '2', '3']]}, '@USE@': {'gold_contexts': [['4', '5']]}}}
    tp, fp, fn = _tp_fp_fn_given_paper(pred_dict_for_paper=pred_dict_for_paper, gold_dict_for_paper=gold_dict_for_paper)
    assert tp == 5 and fp == 0 and  fn == 0

    # diff contexts but same gold sents
    metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    gold_dict_for_paper = {'x': [], 'y': {'@BACK@': {'gold_contexts': [['1'], ['2'], ['3']]}, '@USE@': {'gold_contexts': [['4'], ['5']]}}}
    tp, fp, fn = _tp_fp_fn_given_paper(pred_dict_for_paper=pred_dict_for_paper, gold_dict_for_paper=gold_dict_for_paper)
    assert tp == 5 and fp == 0 and fn == 0

    # pred correct sents, but wrong intents for them
    metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    gold_dict_for_paper = {'x': [], 'y': {'@EXT@': {'gold_contexts': [['1'], ['2'], ['3']]}, '@BACK@': {'gold_contexts': [['4'], ['5']]}}}
    tp, fp, fn = _tp_fp_fn_given_paper(pred_dict_for_paper=pred_dict_for_paper, gold_dict_for_paper=gold_dict_for_paper)
    assert tp == 0 and fp == 5 and fn == 5

    # pred correct intents, but wrong sents for them
    metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    gold_dict_for_paper = {'x': [], 'y': {'@BACK@': {'gold_contexts': [['7'], ['8'], ['9']]}, '@USE@': {'gold_contexts': [['10'], ['11']]}}}
    tp, fp, fn = _tp_fp_fn_given_paper(pred_dict_for_paper=pred_dict_for_paper, gold_dict_for_paper=gold_dict_for_paper)
    assert tp == 0 and fp == 5 and fn == 5

    # pred extra intents
    metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    gold_dict_for_paper = {'x': [], 'y': {'@BACK@': {'gold_contexts': [['1'], ['2'], ['3']]}}}
    tp, fp, fn = _tp_fp_fn_given_paper(pred_dict_for_paper=pred_dict_for_paper, gold_dict_for_paper=gold_dict_for_paper)
    assert tp == 3 and fp == 2 and fn == 0

    # pred missing intents for the same sents
    metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    gold_dict_for_paper = {'x': [], 'y': {'@BACK@': {'gold_contexts': [['1', '2', '3']]}, '@USE@': {'gold_contexts': [['4', '5']]}, '@EXT@': {'gold_contexts': [['1', '2', '3'], ['4', '5']]}}}
    tp, fp, fn = _tp_fp_fn_given_paper(pred_dict_for_paper=pred_dict_for_paper, gold_dict_for_paper=gold_dict_for_paper)
    assert tp == 5 and fp == 0 and fn == 5

    # pred extra sents for same intents
    metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    gold_dict_for_paper = {'x': [], 'y': {'@BACK@': {'gold_contexts': [['1', '2']]}, '@USE@': {'gold_contexts': [['4']]}}}
    tp, fp, fn = _tp_fp_fn_given_paper(pred_dict_for_paper=pred_dict_for_paper, gold_dict_for_paper=gold_dict_for_paper)
    assert tp == 3 and fp == 2 and fn == 0

    # pred missing sents (either same or diff context) for same intents
    metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    gold_dict_for_paper = {'x': [], 'y': {'@BACK@': {'gold_contexts': [['1', '2', '3'], ['4']]}, '@USE@': {'gold_contexts': [['4', '5', '6']]}}}
    tp, fp, fn = _tp_fp_fn_given_paper(pred_dict_for_paper=pred_dict_for_paper, gold_dict_for_paper=gold_dict_for_paper)
    assert tp == 5 and fp == 0 and fn == 2

    # missing preds altogether
    pred_dict_for_paper = {}
    gold_dict_for_paper = {'x': [], 'y': {'@BACK@': {'gold_contexts': [['1', '2', '3']]}, '@USE@': {'gold_contexts': [['4', '5']]}}}
    tp, fp, fn = _tp_fp_fn_given_paper(pred_dict_for_paper=pred_dict_for_paper, gold_dict_for_paper=gold_dict_for_paper)
    assert tp == 0 and fp == 0 and fn == 0


def compute_per_paper_metrics(pred_dict: Dict, gold_dict: Dict) -> Dict:
    paper_id_to_metrics = defaultdict(dict)

    for paper_id, gold_dict_for_paper in gold_dict.items():
        pred_dict_for_paper = pred_dict.get(paper_id, {})
        tp, fp, fn = _tp_fp_fn_given_paper(pred_dict_for_paper=pred_dict_for_paper, gold_dict_for_paper=gold_dict_for_paper)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        paper_id_to_metrics[paper_id]['tp'] = tp
        paper_id_to_metrics[paper_id]['fp'] = fp
        paper_id_to_metrics[paper_id]['fn'] = fn
        paper_id_to_metrics[paper_id]['p'] = p
        paper_id_to_metrics[paper_id]['r'] = r
        paper_id_to_metrics[paper_id]['f1'] = 2 * (p * r) / (p + r) if (p + r) > 0.0 else 0.0
        paper_id_to_metrics[paper_id]['num_gold_sent'] = len({sent for intent, data in gold_dict_for_paper['y'].items() for context in data['gold_contexts'] for sent in context})
        paper_id_to_metrics[paper_id]['num_gold_intent'] = len(gold_dict_for_paper['y'])
        paper_id_to_metrics[paper_id]['num_gold_contexts'] = len([context for intent, data in gold_dict_for_paper['y'].items() for context in data['gold_contexts']])
        paper_id_to_metrics[paper_id]['num_gold_cite_sents'] = len({cite_sent for intent, data in gold_dict_for_paper['y'].items() for cite_sent in data['cite_sentences']})
        paper_id_to_metrics[paper_id]['num_pred_sent'] = len({sent for intent, sents in pred_dict_for_paper.items() for sent in sents})
        paper_id_to_metrics[paper_id]['num_pred_intent'] = len(pred_dict_for_paper)
        paper_id_to_metrics[paper_id]['paper_len'] = len(gold_dict_for_paper['x'])

    return paper_id_to_metrics



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, help='Path to JSON containing predicted sentences')
    parser.add_argument('--gold', type=str, help='Path to JSON containing gold contexts. This is the original dataset.')
    parser.add_argument('--output', type=str, help='Path to an output JSON file.')
    args = parser.parse_args()

    # load pred
    with open(args.pred) as f_in:
        pred_dict = json.load(f_in)

    # load gold
    with open(args.gold) as f_in:
        gold_dict = json.load(f_in)

    # calculate bunch of stuff per paper
    per_paper_metrics = dict(compute_per_paper_metrics(pred_dict=pred_dict, gold_dict=gold_dict))

    # calculate macro metrics
    metric_logger = defaultdict(list)
    for paper_id, paper_metrics in per_paper_metrics.items():
        for metric_name, metric_value in paper_metrics.items():
            metric_logger[metric_name].append(metric_value)
    metrics = {}
    for metric_name, metric_values in metric_logger.items():
        metrics[f'mean-{metric_name}'] = f'{np.mean(metric_values)}'
        metrics[f'std-{metric_name}'] = f'{np.std(metric_values)}'

    # micro metrics
    overall_tp = sum(metric_logger['tp'])
    overall_fp = sum(metric_logger['fp'])
    overall_fn = sum(metric_logger['fn'])
    overall_p = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    overall_r = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    metrics['overall-p'] = overall_p
    metrics['overall-r'] = overall_r
    metrics['overall-f1'] = 2 * (overall_p * overall_r) / (overall_p + overall_r) if (overall_p + overall_r) > 0.0 else 0.0
    metrics['num_papers'] = len(per_paper_metrics)

    # write
    with open(args.output, 'w') as f_out:
        json.dump(metrics, f_out, indent=4)
