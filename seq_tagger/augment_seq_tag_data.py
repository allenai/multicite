"""

This script contains all ways we want to augment the processed sequence tagging data.

* For real world evaluation *

1. We want an instance for every <CITE> marker X every @INTENT@. Easiest thing to do is generate all of these,
   and we can always just filter out the ones that are already represented in the original gold test set.

2. Do we need to generate negative examples? We have negative @INTENT@ per <CITE> naturally from above.
   The negatives we're missing are negative *CONTEXTS*.  There are 2 types:

    a) This is the gold context, but it's truncated due to the window.
    b) This is not even the gold context.

    To handle (a), for every example where window contains gold context, we should generate $K$ windows that clip the gold.
    We don't need to do anything special to handle (b) -- It should just come from step 1.

3. To keep track of these generated instances, give them different IDs so it doesn't get confusing. Start counting from
   a high number, like 30000.



To run:

    gold_data="data/full-v20210918.json"
    for window in 1 3 5 7 9 11
    do
        model="allenai-scibert_scivocab_uncased__${window}__1__07-01-02"
        for fold in 0 1 2 3 4
        do
            python seq_tagger/augment_seq_tag_data.py \
            --original data/${model}/${fold}/ \
            --full ${gold_data} \
            --augmented data/${model}/${fold}-aug/
        done
    done

"""


from tqdm import tqdm
import os
import json
import argparse

import random


try:
    from seq_tagger.utils import instance_id_to_paper_id_and_intent, sent_id_to_pos
    from seq_tagger.const import INTENT_TOKENS, CITE_START, CITE_END, IGNORE_SENTS, WRONG_INTENT_INSTANCE_START_ID
except ImportError:
    from utils import instance_id_to_paper_id_and_intent, sent_id_to_pos
    from const import INTENT_TOKENS, CITE_START, CITE_END, IGNORE_SENTS, WRONG_INTENT_INSTANCE_START_ID


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=str, help='Path to DIR containing processed train|dev|test.jsonl.', default='data/allenai-scibert_scivocab_uncased__11__1__07-01-02/0/')
    parser.add_argument('--full', type=str, help='Path to full original dataset (processing agnostic).', default='data/full-v20210918.json')
    parser.add_argument('--augmented', type=str, help='Path to DIR to output augmented train|dev|test.jsonl.', default='data/allenai-scibert_scivocab_uncased__11__1__07-01-02/0-aug/')
    parser.add_argument('--keep_original', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.augmented, exist_ok=True)

    # load full gold data
    with open(args.full) as f_in:
        full_dict = json.load(f_in)

    # load train|dev|test JSONLs
    for fname in ['train', 'dev', 'test']:
        examples = []
        with open(os.path.join(args.original, f'{fname}.jsonl')) as f_in:
            for line in f_in:
                examples.append(json.loads(line))
        print(f'{len(examples)} original examples in {fname}')

        # augment
        augmented_examples = []
        for example in tqdm(examples):

            # for every example, make an identical window copy of it for all other @INTENT@ where all sents are not-context.
            # this means models need to pay attention to the query @INTENT@.

            paper_id, intent = instance_id_to_paper_id_and_intent(instance_id=example['instance_id'])
            for intent in INTENT_TOKENS:
                if intent not in full_dict[paper_id]['y']:      # avoid accidentally constructing a false negative example
                    augmented_examples.append({
                        'id': WRONG_INTENT_INSTANCE_START_ID + example['id'],
                        'instance_id': f"{example['instance_id']}__wrong_intent",
                        'intent': intent,
                        'sentences': example['sentences'],
                        'labels': ['not-context' for label in example['labels']],
                        'sent_ids': example['sent_ids']
                    })
        print(f'Constructed {len(augmented_examples)} augmented examples')

        # maybe keep original
        if args.keep_original:
            print(f'Keeping original examples in augmentated set')
            augmented_examples.extend(examples)

        # write
        with open(os.path.join(args.augmented, f'{fname}.jsonl'), 'w') as f_out:
            for example in augmented_examples:
                json.dump(example, f_out)
                f_out.write('\n')
