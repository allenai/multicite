"""

Sequence tagging


python seq_tagger/train_seq_tagger.py --input data/allenai-scibert_scivocab_uncased__11__1__07-01-02/0/ --output output/allenai-scibert_scivocab_uncased__11__1__07-01-02/0/

"""

import os
import json

import argparse

from collections import Counter, defaultdict

import torch
import torchmetrics
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.utils import download_url
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from seq_tagger.const import SPECIAL_TOKENS, PAD_TOKEN_ID


class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        all_labels = sorted(set([label for e in self.data for label in e["labels"]]))
        self.label_map = {label: i for i, label in enumerate(all_labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_item = self.data[idx]
        cls_token = self.tokenizer.cls_token
        text = cls_token.join(current_item["sentences"])
        labels = current_item["labels"]
        token_ids = self.tokenizer.encode(text)
        intent_id = dm.tokenizer._convert_token_to_id_with_added_voc(current_item['intent'])
        token_ids = [intent_id] + token_ids

        # TODO: ugly hack to deal with cls tokens that happen after truncation
        # we truncate sequence to length 512 - "num_cls_tokens appearing after 512"
        # then add "num_cls_tokens appearing after 512" to the truncated sequence
        # this results in model underperforming in these situations
        if len(token_ids) > 512:
            cls_indices = [i for i, e in enumerate(token_ids) if e == self.tokenizer.cls_token_id]
            cls_outside_truncation_count = len([idx for idx in cls_indices if idx > 511])
            token_ids = token_ids[: 512 - cls_outside_truncation_count]
            token_ids.extend([self.tokenizer.cls_token_id for _ in range(cls_outside_truncation_count)])
            assert len(labels) == len([e for e in token_ids if e == self.tokenizer.cls_token_id])
        return {
            "text": token_ids,
            "labels": [self.label_map[e] for e in labels],
            'id': [current_item['id'] for _ in labels]
        }

    @staticmethod
    def collate_fn(data):
        instance_ids = [torch.tensor(e["id"]) for e in data]
        token_ids = [torch.tensor(e["text"]) for e in data]
        labels = [torch.tensor(e["labels"]) for e in data]
        instance_ids_tensor = torch.nn.utils.rnn.pad_sequence(instance_ids, batch_first=True, padding_value=0)
        token_ids_tensor = torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=True, padding_value=0)
        labels_tensor = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_ids": token_ids_tensor, "labels": labels_tensor, 'instance_ids': instance_ids_tensor}



class MyDataModule(LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str = "allenai/scibert_scivocab_uncased",
        max_seq_length: int = 512,
        batch_size: int = 8,
        cache_dir: str = "data/allenai-scibert_scivocab_uncased__11__1__07-01-02/0/",
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, add_prefix_space=False, use_fast=True)
        self.tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
        self.cache_dir = cache_dir

    def setup(self, stage = 'fit'):
        def load_data(path):
            data = []
            with open(path) as fin:
                for line in fin:
                    data.append(json.loads(line))
            return data

        if stage == "fit" or stage is None:
            train_data = load_data(os.path.join(self.cache_dir, 'train.jsonl'))
            val_data = load_data(os.path.join(self.cache_dir , 'dev.jsonl'))
            test_data = load_data(os.path.join(self.cache_dir, 'test.jsonl'))
            self.train_dataset = MyDataset(train_data, self.tokenizer)
            self.val_dataset = MyDataset(val_data, self.tokenizer)
            self.test_dataset = MyDataset(test_data, self.tokenizer)

    def prepare_data(self):
        assert os.path.exists(self.cache_dir)
        assert os.path.exists(os.path.join(self.cache_dir, 'train.jsonl'))
        assert os.path.exists(os.path.join(self.cache_dir, 'dev.jsonl'))
        assert os.path.exists(os.path.join(self.cache_dir, 'test.jsonl'))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=MyDataset.collate_fn, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=MyDataset.collate_fn, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=MyDataset.collate_fn, num_workers=0)



class MyTransformer(LightningModule):
    def __init__(
        self,
        tokenizer,
        val_pred_output_path: str,
        test_pred_output_path: str,
        model_name_or_path: str = "allenai/scibert_scivocab_uncased",
        num_labels: int = 2,
        learning_rate: float = 3e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = tokenizer
        self.model = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.metric_acc = torchmetrics.Accuracy()
        self.metric_f1 = torchmetrics.F1(average='macro', num_classes=num_labels)
        self.batch_size = batch_size
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, num_labels)
        self.loss_fn = CrossEntropyLoss()
        self.val_pred_output_path = val_pred_output_path
        self.test_pred_output_path = test_pred_output_path

    def forward(self, **inputs):
        output = self.model(inputs["input_ids"])
        cls_output_state = output["last_hidden_state"][inputs["input_ids"] == self.tokenizer.cls_token_id]
        logits = self.classifier(cls_output_state)
        labels = inputs["labels"][inputs["labels"] != PAD_TOKEN_ID] if "labels" in inputs else None
        instance_ids = inputs["instance_ids"][inputs["labels"] != PAD_TOKEN_ID] if "labels" in inputs else None
        loss = self.loss_fn(logits, labels) if "labels" in inputs else None
        return loss, logits, labels, instance_ids

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        acc = outputs[1]
        self.log("loss", loss)
        self.log("accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits, labels, instance_ids = outputs
        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()
        return {"loss": val_loss, "preds": preds, "labels": labels, 'instance_ids': instance_ids}

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        test_loss, logits, labels, instance_ids = outputs
        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()
        return {"loss": test_loss, "preds": preds, "labels": labels, 'instance_ids': instance_ids}

    def validation_epoch_end(self, outputs):
        ids = torch.cat([x["instance_ids"] for x in outputs]).detach().cpu()
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss)
        val_acc = self.metric_acc(preds, labels)
        val_f1 = self.metric_f1(preds, labels)
        self.log("val_acc", val_acc, prog_bar=True)
        self.log("val_f1", val_f1, prog_bar=True)

        id_to_preds = defaultdict(list)
        for id, pred, label in zip(ids.tolist(), preds.tolist(), labels.tolist()):
            id_to_preds[id].append({'pred': pred, 'label': label})
        with open(self.val_pred_output_path, 'w') as f_out:
            json.dump(dict(id_to_preds), f_out, indent=4)

        return loss

    def test_epoch_end(self, outputs):
        ids = torch.cat([x["instance_ids"] for x in outputs]).detach().cpu()
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("test_loss", loss)
        test_acc = self.metric_acc(preds, labels)
        test_f1 = self.metric_f1(preds, labels)
        self.log("test_acc", test_acc, prog_bar=True)
        self.log("test_f1", test_f1, prog_bar=True)

        id_to_preds = defaultdict(list)
        for id, pred, label in zip(ids.tolist(), preds.tolist(), labels.tolist()):
            id_to_preds[id].append({'pred': pred, 'label': label})
        with open(self.test_pred_output_path, 'w') as f_out:
            json.dump(dict(id_to_preds), f_out, indent=4)

        return loss

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.train_dataloader()
        tb_size = self.hparams.batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Directory path to data files.', required=True)
    parser.add_argument('-o', '--output', type=str, help='Directory path to output files.', required=True)
    parser.add_argument('--model_name_or_path', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--max_steps', type=int)
    args = parser.parse_args()

    dm = MyDataModule(model_name_or_path=args.model_name_or_path, max_seq_length=512, batch_size=args.batch_size, cache_dir=args.input)
    dm.setup()

    # double-check data
    first_example = dm.train_dataset[0]
    sample_text = dm.tokenizer.convert_ids_to_tokens(ids=first_example['text'])
    print(f'First example input text: {sample_text}')
    num_pred_targets = len([token for token in first_example['text'] if token == dm.tokenizer.cls_token_id])
    print(f'Num pred targets ({dm.tokenizer.cls_token}): {num_pred_targets}')
    special_token_ids = [token for token in first_example['text'] if token in dm.tokenizer.additional_special_tokens_ids]
    print(f'Special tokens in input: {dm.tokenizer.convert_ids_to_tokens(ids=special_token_ids)}')
    labels = Counter([label for label in first_example['labels']])
    print(f'Labels ({dm.train_dataset.label_map}): {labels}')

    os.makedirs(args.output, exist_ok=True)
    model = MyTransformer(warmup_steps=args.warmup_steps, tokenizer=dm.tokenizer,
                          val_pred_output_path=os.path.join(args.output, 'val_pred.json'),
                          test_pred_output_path=os.path.join(args.output, 'test_pred.json'))
    trainer = Trainer(gpus=args.gpus, progress_bar_refresh_rate=5, max_epochs=args.max_epochs, max_steps=args.max_steps)
    trainer.fit(model, dm)
    trainer.save_checkpoint(os.path.join(args.output, 'model.ckpt'))
    # test this works
    # new_model = MyTransformer.load_from_checkpoint(checkpoint_path=os.path.join(args.output, 'model.ckpt'))

    val_results = trainer.validate(model, dm.val_dataloader())
    test_results = trainer.test(model, dm.test_dataloader())


