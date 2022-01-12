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
to the original sentences.



To run this script, point it to the top-level directory containing all the CV Folds:

    for window in 1 3 5 7 9 11
    do
      python seq_tagger/collect_organize_preds_across_folds.py \
      --input data/allenai-scibert_scivocab_uncased__${window}__1__07-01-02/ \
      --pred_dirname output/allenai-scibert_scivocab_uncased__${window}__1__07-01-02__batch32/ \
      --pred_fname test-4 \
      --full data/full-v20210918.json \
      --output output/allenai-scibert_scivocab_uncased__${window}__1__07-01-02__batch32/
    done


This should result in new file being created at *OUTPUT*.

"""


from typing import Dict, Tuple, List

import os
import json
import argparse
import re
from collections import defaultdict

from glob import glob


def instance_id_to_paper_id_and_intent(instance_id: str) -> Tuple[str, str]:
    match = re.match(r'(.+)__(@.+@)__[0-9]+', instance_id)
    return match.group(1), match.group(2)


def sent_id_to_pos(sent_id: str) -> int:
    match = re.match(r'.+-C001-([0-9]+)', sent_id)
    return int(match.group(1)) - 1      # in our multicite dataset, sent_ids counts from 1, not 0


def reformat_all_preds_to_dict(all_preds: List[Dict]) -> Dict:
    # Reformats prediction results into
    # {
    #     'ABC_ffcefdc73338187d4a6b2dc2f0bb47_28': {
    #         '@BACK@': ['ffcefdc73338187d4a6b2dc2f0bb47-C001-6', 'ffcefdc73338187d4a6b2dc2f0bb47-C001-27', 'ffcefdc73338187d4a6b2dc2f0bb47-C001-44'],
    #         '@DIF@': ['ffcefdc73338187d4a6b2dc2f0bb47-C001-111'],
    #         ...
    #     },
    #     ...
    # }

    POS_CONTEXT_LABEL = 0

    def _jsonify(extraction_dict) -> Dict:
        return {
            paper_id: {
                intent: sorted(pred_sents, key=lambda s: sent_id_to_pos(sent_id))
                for intent, pred_sents in intent_to_pred_sents.items()
            }
            for paper_id, intent_to_pred_sents in extraction_dict.items()
        }

    paper_id_to_intent_to_pred_sents = defaultdict(lambda: defaultdict(list))
    for result in all_preds:
        id = result['id']
        preds = [p['pred'] for p in result['preds']]                    # list of 1s or 0s
        labels = [p['label'] for p in result['preds']]                  # list of 1s or 0s
        assert len(preds) == len(labels) == len(ID_TO_SENT_IDS[id])     # sanity check. should all be size of window.

        # map back to instance_id and sent_ids
        paper_id, intent = instance_id_to_paper_id_and_intent(instance_id=ID_TO_INSTANCE_ID[id])

        # put them into nested dict
        for pred, label, sent_id in zip(preds, labels, ID_TO_SENT_IDS[id]):
            # prediction of 0 stands for a positive 'context' prediction.
            # also, dont add duplicate sent_ids to context.
            if pred == POS_CONTEXT_LABEL and sent_id not in paper_id_to_intent_to_pred_sents[paper_id][intent]:
                paper_id_to_intent_to_pred_sents[paper_id][intent].append(sent_id)

    # reformat dict
    return _jsonify(paper_id_to_intent_to_pred_sents)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Directory path to cross-fold validation model input files.', default='data/allenai-scibert_scivocab_uncased__11__1__07-01-02/')
    parser.add_argument('--pred_dirname', type=str, help='Directory path to cross-fold model output files.', default='output/allenai-scibert_scivocab_uncased__11__1__07-01-02__batch32/')
    parser.add_argument('--pred_fname', type=str, help='Name of JSONL file containing predictions. Should be same across ALL CV folds.', default='test-4')
    parser.add_argument('--full', type=str, help='Directory path to original full release dataset (processing agnostic).', default='data/full-v20210918.json')
    parser.add_argument('--output', type=str, help='Path to an output directory.', default='results/allenai-scibert_scivocab_uncased__11__1__07-01-02/')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # load the window-processed inputs to the model so that we can map the predictions back to original instances & sentences.
    # NOTE: this wont be comprehensive to every possible instance in Multicite dataset, due to window size processing.
    ID_TO_INSTANCE_ID = {}
    ID_TO_SENT_IDS = {}
    with open(os.path.join(args.input, 'full.json')) as f_in:
        all_examples = json.load(f_in)['data']
        for example in all_examples:
            id = example['id']
            instance_id = example['instance_id']
            sent_ids = example['sent_ids']
            ID_TO_INSTANCE_ID[id] = instance_id
            ID_TO_SENT_IDS[id] = sent_ids
    print(f'Loaded {len(all_examples)} examples and organized by ID')


    # load model predictions across all the folds
    all_preds = []
    for infile in glob(os.path.join(args.pred_dirname, '*/', f'{args.pred_fname}.jsonl')):
        with open(infile) as f_in:
            for line in f_in:
                pred = json.loads(line)
                all_preds.append(pred)

    # organize them into predicted contexts (per document)
    paper_id_to_intent_to_pred_sents = reformat_all_preds_to_dict(all_preds=all_preds)

    print(f'Saved {len(paper_id_to_intent_to_pred_sents)} papers worth of predictions.')
    with open(os.path.join(args.output, f'all_preds_for_{args.pred_fname}.json'), 'w') as f_out:
        json.dump(paper_id_to_intent_to_pred_sents, f_out, indent=4)
