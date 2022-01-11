"""

Training should generate useful files like:

|-- allenai-scibert_scivocab_uncased__3__1__07-01-02__batch32/
    |-- 0/
        # predictions
        |-- val-0.jsonl     # per epoch
        |-- val-1.jsonl
        |-- val-2.jsonl
        |-- val-3.jsonl
        |-- val-4.jsonl
        |-- test-4.jsonl    # at the end, test set
        # checkpoints
        |-- epoch=00-step=268-val_loss=0.3514-val_f1=0.8505.ckpt
        |-- epoch=01-step=537-val_loss=0.3514-val_f1=0.8505.ckpt
        |-- ...
        # metrics
        |-- val-metrics0.json
        |-- val-metrics1.json
        |-- val-metrics2.json
        |-- val-metrics3.json
        |-- val-metrics4.json
        |-- test-metrics4.json
    |-- 1/
    |-- 2/
    |-- 3/
    |-- 4/                  # per fold


Those metrics unfortunately aren't comparable across Window sizes (because the test data is different depending on processing).

As such, we want to do some evaluation that's fair across the models.  For each test set predictions, we'll map those back
to the original sentences.  And use those to generate some F1 metrics.

Note: These F1 metrics will be ORACLE scores because the test set predictions were done on inputs with Gold INTENT
and Gold CONTEXT.  This script will only penalize models for if window is too small -->
    then the model technically didn't predict anything & therefore will have lower recall.


To run this script, point it to the top-level directory containing all the CV Folds:

python eval_seq_tagger_preds.py \
    --input /net/nfs2.s2-research/kylel/multicite-2022/data/allenai-scibert_scivocab_uncased__5__1__07-01-02/ \
    --pred_dirname /net/nfs2.s2-research/kylel/multicite-2022/output/allenai-scibert_scivocab_uncased__5__1__07-01-02__batch32/ \
    --pred_fname test-4 \
    --full /net/nfs2.s2-research/kylel/multicite-2022/data/full-v20210918.json \
    --output /net/nfs2.s2-research/kylel/multicite-2022/results/allenai-scibert_scivocab_uncased__5__1__07-01-02__batch32/ \


This should result in new files being created:

|-- allenai-scibert_scivocab_uncased__3__1__07-01-02__batch32/
    |-- all_preds_for_test-4.jsonl
    |-- per_paper_metrics_for_test-4.csv
    |--


"""

from typing import Dict, Tuple, List

import os
import json
import argparse
import re
from collections import defaultdict

from glob import glob

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def instance_id_to_paper_id_and_intent(instance_id: str) -> Tuple[str, str]:
    match = re.match(r'(.+)__(@.+@)__[0-9]+', instance_id)
    return match.group(1), match.group(2)

def sent_id_to_pos(sent_id: str) -> int:
    match = re.match(r'.+-C001-([0-9]+)', sent_id)
    return int(match.group(1)) - 1      # in our multicite dataset, sent_ids counts from 1, not 0

def compute_paper_scores(pred_sent_ids: List[str], gold_sent_ids: List[str], paper_len: int) -> Dict:

    # convert the gold & pred extractions to a vector of [0, 1, 0, 0, ...]
    if gold_sent_ids:
        _g = {sent_id_to_pos(g) for g in gold_sent_ids}
        y_true = [1 if i in _g else 0 for i in range(paper_len)]
    else:
        y_true = [0 for _ in range(paper_len)]

    if pred_sent_ids:
        _p = {sent_id_to_pos(p) for p in pred_sent_ids}
        y_pred = [1 if i in _p else 0 for i in range(paper_len)]
    else:
        y_pred = [0 for _ in range(paper_len)]

    # use sklearn to compute basic metrics
    metrics = {}
    p, r, f1, support = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    metrics['tp'] = tp
    metrics['tn'] = tn
    metrics['fp'] = fp
    metrics['fn'] = fn
    metrics['p'] = p[-1]
    metrics['r'] = r[-1]
    metrics['f1'] = f1[-1]
    metrics['support'] = {'not-context': support[0], 'context': support[-1], 'total': support.sum()}

    return metrics


def _jsonify_extraction_dict(extraction_dict) -> Dict:
    return {
        paper_id: {
            intent: sorted(pred_sents, key=lambda s: sent_id_to_pos(sent_id))
            for intent, pred_sents in intent_to_pred_sents.items()
        }
        for paper_id, intent_to_pred_sents in extraction_dict.items()
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Directory path to cross-fold validation model input files.', default='data/allenai-scibert_scivocab_uncased__11__1__07-01-02/')
    parser.add_argument('--pred_dirname', type=str, help='Directory path to cross-fold model output files.', default='output/allenai-scibert_scivocab_uncased__11__1__07-01-02/')
    parser.add_argument('--pred_fname', type=str, help='Name of JSONL file containing predictions. Should be same across ALL CV folds.', default='test-4')
    parser.add_argument('--full', type=str, help='Directory path to original full release dataset (processing agnostic).', default='data/full-v20210918.json')
    parser.add_argument('--output', type=str, help='Path to an output directory.', default='output/allenai-scibert_scivocab_uncased__11__1__07-01-02/')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)


    # load the window-processed inputs to the model so that we can map the predictions back to original instances & sentences.
    # NOTE: this wont be comprehensive to every possible instance in Multicite dataset, due to window size processing.
    id_to_instance_id = {}
    id_to_sent_ids = {}
    with open(os.path.join(args.input, 'full.json')) as f_in:
        all_examples = json.load(f_in)['data']
        for example in all_examples:
            id = example['id']
            instance_id = example['instance_id']
            sent_ids = example['sent_ids']
            id_to_instance_id[id] = instance_id
            id_to_sent_ids[id] = sent_ids
    print(f'Loaded {len(all_examples)} examples and organized by ID')


    # load model predictions across all the folds & then organize them into predicted contexts (per document). for example:
    # {
    #     'ABC_ffcefdc73338187d4a6b2dc2f0bb47_28': {
    #         '@BACK@': ['ffcefdc73338187d4a6b2dc2f0bb47-C001-6', 'ffcefdc73338187d4a6b2dc2f0bb47-C001-27', 'ffcefdc73338187d4a6b2dc2f0bb47-C001-44'],
    #         '@DIF@': ['ffcefdc73338187d4a6b2dc2f0bb47-C001-111'],
    #         ...
    #     },
    #     ...
    # }
    paper_id_to_intent_to_pred_sents = defaultdict(lambda: defaultdict(list))
    for infile in glob(os.path.join(args.pred_dirname, '*/', f'{args.pred_fname}.jsonl')):
        print(f'Processing {infile}...')
        with open(infile) as f_in:
            for line in f_in:
                result = json.loads(line)
                # pull out data
                id = result['id']
                preds = [p['pred'] for p in result['preds']]                    # list of 1s or 0s
                labels = [p['label'] for p in result['preds']]                  # list of 1s or 0s
                assert len(preds) == len(labels) == len(id_to_sent_ids[id])     # sanity check. should all be size of window.
                # map back to instance_id and sent_ids
                paper_id, intent = instance_id_to_paper_id_and_intent(instance_id=id_to_instance_id[id])
                for pred, label, sent_id in zip(preds, labels, id_to_sent_ids[id]):
                    # prediction of 0 stands for a positive 'context' prediction.
                    # also, dont add duplicate sent_ids to context.
                    if pred == 0 and sent_id not in paper_id_to_intent_to_pred_sents[paper_id][intent]:
                        paper_id_to_intent_to_pred_sents[paper_id][intent].append(sent_id)

    print(f'Saved {len(paper_id_to_intent_to_pred_sents)} papers worth of predictions.')
    with open(os.path.join(args.output, f'all_preds_for_{args.pred_fname}.json'), 'w') as f_out:
        json.dump(_jsonify_extraction_dict(paper_id_to_intent_to_pred_sents), f_out, indent=4)


    # finally, compare the predicted extractions to original contexts.
    paper_id_to_metrics = {}
    with open(args.full) as f_in:
        full = json.load(f_in)
        for paper_id, data in full.items():
            paper_len = len(data['x'])
            for intent, annotations in data['y'].items():
                # original contexts stored in a structured way. just unfurl them into a single set of sent_ids for this eval
                gold_sent_ids = sorted({sent_id for context in annotations['gold_contexts'] for sent_id in context}, key=lambda s: sent_id_to_pos(sent_id))
                if paper_id in paper_id_to_intent_to_pred_sents and intent in paper_id_to_intent_to_pred_sents[paper_id]:
                    pred_sent_ids = paper_id_to_intent_to_pred_sents[paper_id][intent]
                else:
                    # in cases where window is too small, there may be no (oracle) example that results in a prediction. default to No Prediction.
                    pred_sent_ids = []
                metrics = compute_paper_scores(pred_sent_ids=pred_sent_ids, gold_sent_ids=gold_sent_ids, paper_len=paper_len)
                paper_id_to_metrics[paper_id] = metrics



    # write scores
    with open(os.path.join(args.output, f'per_paper_metrics_for_{args.pred_fname}.csv'), 'w') as f_out:
        f_out.write(','.join(['paper_id', 'tp', 'tn', 'fp', 'fn', 'p', 'r', 'f1', 'num_context', 'num_not_context', 'n']))
        f_out.write('\n')
        for paper_id, scores in paper_id_to_metrics.items():
            f_out.write(','.join([
                paper_id,
                f"{scores['tp']}",
                f"{scores['tn']}",
                f"{scores['fp']}",
                f"{scores['fn']}",
                f"{scores['p']}",
                f"{scores['r']}",
                f"{scores['f1']}",
                f"{scores['support']['context']}",
                f"{scores['support']['not-context']}",
                f"{scores['support']['total']}"
            ]))
            f_out.write('\n')
