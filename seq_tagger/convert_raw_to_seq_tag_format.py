"""

Converts raw data to sequence tagging format  (see data_format.jsonl)


python seq_tagger/convert_raw_to_seq_tag_format.py --input data/full-v20210918.json --output data/ --model allenai/scibert_scivocab_uncased --window 1 --train-prob 0.7 --dev-prob 0.1 --test-prob 0.2
python seq_tagger/convert_raw_to_seq_tag_format.py --input data/full-v20210918.json --output data/ --model allenai/scibert_scivocab_uncased --window 3 --train-prob 0.7 --dev-prob 0.1 --test-prob 0.2
python seq_tagger/convert_raw_to_seq_tag_format.py --input data/full-v20210918.json --output data/ --model allenai/scibert_scivocab_uncased --window 5 --train-prob 0.7 --dev-prob 0.1 --test-prob 0.2
python seq_tagger/convert_raw_to_seq_tag_format.py --input data/full-v20210918.json --output data/ --model allenai/scibert_scivocab_uncased --window 7 --train-prob 0.7 --dev-prob 0.1 --test-prob 0.2
python seq_tagger/convert_raw_to_seq_tag_format.py --input data/full-v20210918.json --output data/ --model allenai/scibert_scivocab_uncased --window 9 --train-prob 0.7 --dev-prob 0.1 --test-prob 0.2
python seq_tagger/convert_raw_to_seq_tag_format.py --input data/full-v20210918.json --output data/ --model allenai/scibert_scivocab_uncased --window 11 --train-prob 0.7 --dev-prob 0.1 --test-prob 0.2


"""


from typing import List, Optional, Dict

import argparse

import os
import random
import json

from tqdm import tqdm
from transformers import AutoTokenizer

from collections import Counter

from seq_tagger.const import SPECIAL_TOKENS, IGNORE_SENTS, CITE_START, CITE_END


def remove_html(sentence: str) -> str:
    cite_start = '<span style="background: yellow; display: inline-block">'
    cite_end = '</span>'
    return sentence.replace(cite_start, '').replace(cite_end, '')


def replace_html(sentence: str, new_cite_start: str, new_cite_end: str) -> str:
    cite_start = '<span style="background: yellow; display: inline-block">'
    cite_end = '</span>'
    return sentence.replace(cite_start, new_cite_start).replace(cite_end, new_cite_end)


def compute_num_tokens(sentence: str, tokenizer):
    encodings_from_sent = tokenizer(
        sentence,
        is_split_into_words=False,
        add_special_tokens=False
    )
    return len(encodings_from_sent['input_ids'])



def split_examples(examples: list, train_prob: float, dev_prob: float, test_prob: float):
    """
    output looks like:
    {
        0: {'train': [...], 'dev': [...], 'test': [...]},
        1: {'train': [...], 'dev': [...], 'test': [...]},
        ...
        k: {'train': [...], 'dev': [...], 'test': [...]},
    }
    """
    assert train_prob + dev_prob + test_prob == 1.0
    num_folds = int(1.0 / test_prob)
    assert 1.0 / test_prob == num_folds  # check that we can evenly divide into folds

    paper_ids = list({example['instance_id'].split('_')[1] for example in examples})
    random.shuffle(paper_ids)

    num_test = int(len(paper_ids) / num_folds)
    num_train = int((len(paper_ids) - num_test) * train_prob / (train_prob + dev_prob))
    num_dev = len(paper_ids) - num_test - num_train
    assert num_train + num_dev + num_test == len(paper_ids)

    # precompute the CV paper ID assignments
    fold_to_paper_ids = {}
    for fold in range(num_folds):
        start_test = fold * num_test
        end_test = start_test + num_test
        test = paper_ids[start_test:end_test]

        remaining_paper_ids = [paper_id for paper_id in paper_ids if paper_id not in test]
        random.shuffle(remaining_paper_ids)
        train = remaining_paper_ids[:num_train]
        dev = remaining_paper_ids[num_train:]
        assert len(train) + len(dev) + len(test) == len(paper_ids)
        fold_to_paper_ids[fold] = (train, dev, test)


    # split examples s.t. all examples from same paper go to same split
    fold_to_examples = {}
    for fold, (train, dev, test) in fold_to_paper_ids.items():
        fold_to_examples[fold] = {
            'train': [],
            'dev': [],
            'test': []
        }
        for example in examples:
            paper_id = example['instance_id'].split('_')[1]
            if paper_id in train:
                fold_to_examples[fold]['train'].append(example)
            elif paper_id in dev:
                fold_to_examples[fold]['dev'].append(example)
            else:
                fold_to_examples[fold]['test'].append(example)
    return fold_to_examples



def build_window(gold_context_sent_pos: List[int],                                                  # annotations
                 sent_pos_to_num_tokens: List[int],                                                 # resources
                 max_num_sents: int, paper_num_sents: int, max_model_num_tokens: int                # constraints
                 ) -> Optional[List[int]]:

    # let's always include the full gold context, at minimum (unless it truncates due to window, in which case we skip)
    gold_context_num_tokens = sum([sent_pos_to_num_tokens[gold_sent_pos] for gold_sent_pos in gold_context_sent_pos])

    # validate our constraints before getting the actual sentence text
    if len(gold_context_sent_pos) > max_num_sents:  # cant even contain the gold context
        print(f'Skipping... Sent Gold: {len(gold_context_sent_pos)}')
        return None
    elif gold_context_num_tokens > max_model_num_tokens:  # too long for even this longformer
        print(f'Skipping... Token Gold: {gold_context_num_tokens}')
        return None
    else:
        """Good to go :)"""

    # now let's randomly append sentences to front/back
    window_sent_pos = [sent_pos for sent_pos in gold_context_sent_pos]
    window_num_tokens = gold_context_num_tokens
    while len(window_sent_pos) <= max_num_sents and window_num_tokens <= max_model_num_tokens:
        before_sent_id = min(window_sent_pos) - 1
        after_sent_id = max(window_sent_pos) + 1

        # if only one valid option, just pick that one
        if before_sent_id < 0 and after_sent_id < paper_num_sents:
            new_sent_id = after_sent_id
        elif before_sent_id >= 0 and after_sent_id >= paper_num_sents:
            new_sent_id = before_sent_id
        # if no valid options, just get out of the whole thing
        elif before_sent_id < 0 and after_sent_id >= paper_num_sents:
            break
        # if both valid options, random
        else:
            choice = random.randint(0, 1)
            if choice == 0:
                new_sent_id = after_sent_id
            else:
                new_sent_id = before_sent_id

        # token length check
        new_sent_num_tokens = sent_pos_to_num_tokens[new_sent_id]
        if window_num_tokens + new_sent_num_tokens <= max_model_num_tokens:

            # now add to window
            window_sent_pos.append(new_sent_id)
            window_num_tokens += sent_pos_to_num_tokens[new_sent_id]

        else:
            # ran out of space so end adding phase
            break

    window_sent_pos = sorted(window_sent_pos)
    return window_sent_pos


def _describe_window(window_sent_pos: List[int], window_num_tokens: int,
                     cite_sent_pos: List[int],
                     gold_context_sent_pos: List[int], gold_context_num_tokens: int):

    where_are_cite_in_window = [window_sent_pos.index(i) for i in cite_sent_pos]
    print(f'Token Window:{window_num_tokens}\tToken Gold:{gold_context_num_tokens}\tSent Window: {len(window_sent_pos)}\tSent Gold: {len(gold_context_sent_pos)}\tCiteInWindow: {where_are_cite_in_window}')


def build_all_examples(raw_data_dict: Dict) -> List[Dict]:

    n = 0
    all_examples = []
    for paper_id, data in tqdm(raw_data_dict.items()):
        paper = data['x']                          # paper
        intent_to_annotations = data['y']          # intent & context annotations

        # 0) clean out IGNORED SENTS
        clean_paper = [sent_dict for sent_dict in paper if sent_dict['text'] not in IGNORE_SENTS]

        # 1) build an easy lookup for sentences from its ID (str) to its list position (int)
        sent_id_to_list_pos: Dict[str, int] = {sent_dict['sent_id']: i for i, sent_dict in enumerate(clean_paper)}
        sent_pos_to_sent_id: Dict[int, str] = {sent_pos: sent_id for sent_id, sent_pos in sent_id_to_list_pos.items()}

        # 2) precompute tokens per sentence (assuming they're cleaned)
        sent_pos_to_clean_text = [replace_html(sent_dict['text'], new_cite_start=CITE_START, new_cite_end=CITE_END) for sent_dict in clean_paper]
        sent_pos_to_num_tokens: List[int] = [compute_num_tokens(clean_sent, tokenizer) for clean_sent in sent_pos_to_clean_text]

        # 3) create a training example out of each gold context (i.e. each cite mention)
        #    note, this code is a bit annoying because we want to convert all sent_ids to positions first
        for intent, annotations in intent_to_annotations.items():
            for idx_gold_context, gold_context_sent_ids in enumerate(annotations['gold_contexts']):

                # 3a) build valid window
                gold_context_sent_pos = [sent_id_to_list_pos[sent_id] for sent_id in gold_context_sent_ids if sent_id in sent_id_to_list_pos]   # ignores if annotated an ignored sent
                window_sent_pos: Optional[List[int]] = build_window(gold_context_sent_pos=gold_context_sent_pos,
                                                                    sent_pos_to_num_tokens=sent_pos_to_num_tokens,
                                                                    max_num_sents=int(args.window),
                                                                    paper_num_sents=len(clean_paper),
                                                                    max_model_num_tokens=500)   # conservatively stay within 512
                if window_sent_pos is None:
                    continue

                # verify there is at least one <CITE>
                cite_sent_pos = [sent_id_to_list_pos[cite_sent_id] for cite_sent_id in annotations['cite_sentences'] if cite_sent_id in gold_context_sent_ids]
                if not cite_sent_pos:
                    continue

                _describe_window(window_sent_pos=window_sent_pos,
                                 window_num_tokens=sum([sent_pos_to_num_tokens[pos] for pos in window_sent_pos]),
                                 cite_sent_pos=cite_sent_pos,
                                 gold_context_sent_pos=gold_context_sent_pos,
                                 gold_context_num_tokens=sum([sent_pos_to_num_tokens[pos] for pos in gold_context_sent_pos]))

                # 3b) get the text and labels
                example = {
                    'id': n,
                    'instance_id': f'{paper_id}__{intent}__{idx_gold_context}',
                    'intent': intent,
                    'sentences': [],
                    'labels': [],
                    'sent_ids': []
                }
                for w_pos in window_sent_pos:
                    example['sentences'].append(sent_pos_to_clean_text[w_pos])
                    example['labels'].append('context' if w_pos in gold_context_sent_pos else 'not-context')
                    example['sent_ids'].append(sent_pos_to_sent_id[w_pos])

                all_examples.append(example)
                n += 1

    return all_examples




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processes raw data to model input format for specified model, or creates a train/dev/test split on processed data.')
    parser.add_argument('-i', '--input', type=str, help='Directory path to input file.', required=True)
    parser.add_argument('-o', '--output', type=str, help='Directory path to output file.', required=True)

    parser.add_argument('-m', '--model', type=str, help='bert-base-uncased or allenai/scibert_scivocab_uncased')
    parser.add_argument('-w', '--window', type=str, help='Can set to `max` or some integer for a maximum window size. Default is max', default='max')
    parser.add_argument('-r', '--rand-state', type=int, help='Integer used to initialized the random state to split the data', default=1)

    parser.add_argument('--train-prob', type=float)
    parser.add_argument('--dev-prob', type=float)
    parser.add_argument('--test-prob', type=float)

    args = parser.parse_args()

    # stuff need from model
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=False, use_fast=True)
    tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})

    # stuff for input
    with open(args.input) as f_in:
        d = json.load(f_in)

    # build examples
    random.seed(args.rand_state)
    all_examples = build_all_examples(raw_data_dict=d)


    # stuff for output
    outdir = os.path.join(args.output, f'{args.model}__{args.window}__{args.rand_state}__{args.train_prob}-{args.dev_prob}-{args.test_prob}'.replace('/', '-').replace('.', ''))
    os.makedirs(outdir, exist_ok=True)
    out_file = os.path.join(outdir, 'full.json')
    with open(out_file, 'w') as outfile:
        json.dump({'data': all_examples}, outfile, indent=4)

    # now do the splitting & log it
    logfile = os.path.join(outdir, 'log.txt')
    with open(logfile, 'w') as f_log:

        fold_to_splits = split_examples(examples=all_examples, train_prob=args.train_prob, dev_prob=args.dev_prob, test_prob=args.test_prob)
        for fold, splits in fold_to_splits.items():
            folddir = os.path.join(outdir, f'{fold}/')
            os.makedirs(folddir, exist_ok=True)
            trainfile = os.path.join(folddir, 'train.jsonl')
            devfile = os.path.join(folddir, 'dev.jsonl')
            testfile = os.path.join(folddir, 'test.jsonl')
            train = splits['train']
            dev = splits['dev']
            test = splits['test']
            with open(trainfile, 'w') as f_train:
                for e in train:
                    json.dump(e, f_train)
                    f_train.write('\n')
            with open(devfile, 'w') as f_val:
                for e in dev:
                    json.dump(e, f_val)
                    f_val.write('\n')
            with open(testfile, 'w') as f_test:
                for e in test:
                    json.dump(e, f_test)
                    f_test.write('\n')

            f_log.write(f'Counting rare class in Fold {fold} \n')
            f_log.write('train ' + f"{Counter([sum([1 if tag == 'context' else 0 for tag in e['labels']]) for e in train])} / {len(train)}\n")
            f_log.write('dev ' + f"{Counter([sum([1 if tag == 'context' else 0 for tag in e['labels']]) for e in dev])} / {len(dev)}\n")
            f_log.write('test ' + f"{Counter([sum([1 if tag == 'context' else 0 for tag in e['labels']]) for e in test])} / {len(test)}\n")
            f_log.write('\n\n')
