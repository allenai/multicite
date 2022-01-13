"""

Loads model checkpoint and runs predictions on a given dataset.


Pretty hacky script but PTL doesn't have good documentation on how to use its models for prediction. This seems easiest.
Just note.  The hackiest part of this is that we're pointing to an input directory containing train|val|test.jsonl
but we're only using test.jsonl for predictions.



Usage?

For example, let's use this to get the predictions for every epoch on the test data (as opposed to just epoch 4):

    data_dir="/net/nfs2.corp/s2-research/kylel/multicite-2022/data"
    ckpt_dir="/net/nfs2.corp/s2-research/kylel/multicite-2022/output"
    output_dir="/net/nfs2.corp/s2-research/kylel/multicite-2022/test_preds"
    for window in 1 3 5 7 9 11
    do
        for fold in 0 1 2 3 4
        do
            for epoch in 0 1 2 3
            do
              temp_dir="${output_dir}/allenai-scibert_scivocab_uncased__${window}__1__07-01-02__batch32/${fold}/${epoch}/"
              ckpt="${ckpt_dir}/allenai-scibert_scivocab_uncased__${window}__1__07-01-02__batch32/${fold}/epoch=0${epoch}*.ckpt"
              python load_ckpt_and_predict.py \
              --input ${data_dir}/allenai-scibert_scivocab_uncased__${window}__1__07-01-02/${fold}/ \
              --ckpt ${ckpt} \
              --output ${temp_dir}/allenai-scibert_scivocab_uncased__${window}__1__07-01-02__batch32/${fold}/${epoch}/ \
              --batch_size 32 \
              --gpus 1 \
              --use_intent

              mv ${temp_dir}/allenai-scibert_scivocab_uncased__${window}__1__07-01-02__batch32/${fold}/${epoch}/test-0.jsonl \
              ${output_dir}/allenai-scibert_scivocab_uncased__${window}__1__07-01-02__batch32/${fold}/test-${epoch}.jsonl

              mv ${temp_dir}/allenai-scibert_scivocab_uncased__${window}__1__07-01-02__batch32/${fold}/${epoch}/test-metrics0.json \
              ${output_dir}/allenai-scibert_scivocab_uncased__${window}__1__07-01-02__batch32/${fold}/test-metrics${epoch}.json

              rmdir ${temp_dir}
            done
        done
    done



which will produce output like:

|-- test_output/
    |-- allenai-scibert_scivocab_uncased__1__1__07-01-02__batch32/
        |-- 0                               # fold
            |-- 0                           # batch
                |-- test-0.jsonl            # predictions
                |-- test-metrics0.json      # not-quite-usable metrics
            |-- 1
            |-- 2
        |-- 1
        |-- 2
    |-- allenai-scibert_scivocab_uncased__3__1__07-01-02__batch32/
    |-- allenai-scibert_scivocab_uncased__5__1__07-01-02__batch32/



*NOTE* you'll need to adapt the commands in this script to handle the __nointent__ cases.




"""

import os
import argparse
import json

from pytorch_lightning import Trainer

try:
    from seq_tagger.train_seq_tagger import MyTransformer, MyDataModule, MyDataset
except ImportError:
    from train_seq_tagger import MyTransformer, MyDataModule, MyDataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Path to data directory.', required=True)
    parser.add_argument('-c', '--ckpt', type=str, help='Path to model ckpt file.', required=True)
    parser.add_argument('-o', '--output', type=str, help='Path to directory for output.', required=True)
    parser.add_argument('--model_name_or_path', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--use_intent', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # load model
    model = MyTransformer.load_from_checkpoint(checkpoint_path=args.ckpt)
    model.val_pred_output_path = args.output
    model.test_pred_output_path = args.output       # hacky, but it works :/

    # load data using PTL Data module. Also not necessary, but keeps things short.
    dm = MyDataModule(model_name_or_path=args.model_name_or_path,
                      max_seq_length=512,
                      batch_size=args.batch_size,
                      cache_dir=args.input,
                      use_intent=args.use_intent)
    dm.setup()

    # use PTL's trainer to avoid writing predition loop. also hacky, but keeps code short.
    trainer = Trainer(gpus=args.gpus, progress_bar_refresh_rate=5, max_steps=0, limit_val_batches=0)

    # do the prediction
    results = trainer.test(model, dm.test_dataloader())
    assert os.path.exists(os.path.join(args.output, 'test-0.jsonl'))        # it's called test-0.jsonl cuz of how DM is set up :/


