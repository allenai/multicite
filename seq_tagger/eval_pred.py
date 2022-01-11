"""

After training, get processing-agnostic test metrics.


python eval_pred.py \
    --input /net/nfs2.s2-research/kylel/multicite-2022/data/allenai-scibert_scivocab_uncased__5__1__07-01-02/ \
    --pred /net/nfs2.s2-research/kylel/multicite-2022/output/allenai-scibert_scivocab_uncased__5__1__07-01-02__batch32/ \
    --full /net/nfs2.s2-research/kylel/multicite-2022/data/full-v20210918.json \
    --result /net/nfs2.s2-research/kylel/multicite-2022/output/allenai-scibert_scivocab_uncased__5__1__07-01-02__batch32/ \


"""

from typing import Dict, Tuple, List

import os
import json
import argparse
import re
from collections import defaultdict

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def instance_id_to_paper_id_and_intent(instance_id: str) -> Tuple[str, str]:
    match = re.match(r'(.+)__(@.+@)__[0-9]+', instance_id)
    return match.group(1), match.group(2)

def sent_id_to_pos(sent_id: str) -> int:
    match = re.match(r'.+-C001-([0-9]+)', sent_id)
    return int(match.group(1))

def compute_paper_scores(pred_sent_ids: List[str], gold_sent_ids: List[str], paper_len: int) -> Dict:
    _g = {sent_id_to_pos(g) for g in gold_sent_ids}
    y_true = [1 if i in _g else 0 for i in range(paper_len)]

    if pred_sent_ids:
        _p = {sent_id_to_pos(p) for p in pred_sent_ids}
        y_pred = [1 if i in _p else 0 for i in range(paper_len)]
    else:
        y_pred = [0 for _ in range(paper_len)]

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Directory path to cross-fold validation INPUT files.', default='data/allenai-scibert_scivocab_uncased__11__1__07-01-02/')
    parser.add_argument('-p', '--pred', type=str, help='Directory path to cross-fold validation PRED files.', default='output/allenai-scibert_scivocab_uncased__11__1__07-01-02/')
    parser.add_argument('-f', '--full', type=str, help='Directory path to original full release dataset (processing agnostic).', default='data/full-v20210918.json')
    parser.add_argument('-r', '--result', type=str, help='Path to an output directory.', default='output/allenai-scibert_scivocab_uncased__11__1__07-01-02/')
    args = parser.parse_args()


    # load the inputs to the model so that we can map the predictions back to original instances & sentences.
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


    # load predictions across all the folds & then organize them into predicted contexts (per document). for example:
    # {
    #     'ABC_ffcefdc73338187d4a6b2dc2f0bb47_28': {
    #         '@BACK@': ['ffcefdc73338187d4a6b2dc2f0bb47-C001-6', 'ffcefdc73338187d4a6b2dc2f0bb47-C001-27', 'ffcefdc73338187d4a6b2dc2f0bb47-C001-44'],
    #         '@DIF@': ['ffcefdc73338187d4a6b2dc2f0bb47-C001-111'],
    #         ...
    #     },
    #     ...
    # }
    paper_id_to_intent_to_pred_sents = defaultdict(lambda: defaultdict(list))
    for fold in os.listdir(args.pred):
        if fold.isnumeric() and os.path.isdir(os.path.join(args.pred, f'{fold}')):
            print(f'Processing fold {fold} in {args.pred}...')
            for fname in os.listdir(os.path.join(args.pred, f'{fold}')):
                if 'test' in fname and 'metrics' not in fname:              # TODO: super hack, but eh...
                    with open(os.path.join(args.pred, f'{fold}', fname)) as f_in:
                        for line in f_in:
                            result = json.loads(line)
                            id = result['id']
                            preds = [p['pred'] for p in result['preds']]
                            labels = [p['label'] for p in result['preds']]
                            assert len(preds) == len(labels) == len(id_to_sent_ids[id])

                            # map back to instance_id and sent_ids
                            paper_id, intent = instance_id_to_paper_id_and_intent(instance_id=id_to_instance_id[id])
                            for pred, label, sent_id in zip(preds, labels, id_to_sent_ids[id]):
                                # 0 = predicted a context label
                                if pred == 0 and sent_id not in paper_id_to_intent_to_pred_sents[paper_id][intent]:
                                    paper_id_to_intent_to_pred_sents[paper_id][intent].append(sent_id)
    paper_id_to_intent_to_pred_sents = {
        paper_id: {
            intent: sorted(pred_sents, key=lambda s: sent_id_to_pos(sent_id))
            for intent, pred_sents in intent_to_pred_sents.items()
        }
        for paper_id, intent_to_pred_sents in paper_id_to_intent_to_pred_sents.items()
    }
    print(f'Saved {len(paper_id_to_intent_to_pred_sents)} papers worth of predictions.')
    with open(os.path.join(args.result, 'all_test_preds.json'), 'w') as f_out:
        json.dump(paper_id_to_intent_to_pred_sents, f_out, indent=4)


    # finally, compare the preds to original
    paper_id_to_scores = {}
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
                paper_id_to_scores[paper_id] = metrics

    # write scores
    with open(os.path.join(args.result, 'per_paper_metrics.csv'), 'w') as f_out:
        f_out.write(','.join(['paper_id', 'tp', 'tn', 'fp', 'fn', 'p', 'r', 'f1', 'num_context', 'num_not_context', 'n']))
        f_out.write('\n')
        for paper_id, scores in paper_id_to_scores.items():
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
