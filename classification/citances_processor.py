import glob
import codecs
import numpy as np
import random

""" Citances processors and helpers """

import logging
from typing import List, Optional, Union
import json
import jsonlines

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import DataProcessor, InputExample, InputFeatures

logger = logging.getLogger(__name__)


def citances_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    return _convert_examples_to_features(
        examples, tokenizer, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode
    )


def _convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = citances_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = citances_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    if label_list is not None:
        label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float]:
        if output_mode == "classification":
            return label_map[example.label]
        if output_mode == "multilabel_classification":
            return [label_map[l] for l in example.label]
        raise KeyError(output_mode)

    if label_list is not None:
        labels = [label_from_example(example) for example in examples]
        #labels = [example.label for example in examples]
    # This is the "standard way" of encoding the input
    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        if label_list is not None:
            feature = InputFeatures(**inputs, label=labels[i])
        else:
            feature = InputFeatures(**inputs)
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


class OurClassificationProcessor(DataProcessor):
    """Processor for the classification version of our data set."""
    def __init__(self, k=None):
        super()
        self.k = k

    def _read_json(self, path):
        with open(path) as f_in:
            return json.load(f_in)

    def get_train_examples(self, path=""):
        """See base class."""
        path += "/train.json"
        logger.info("LOOKING AT {}".format(path))
        if self.k is None:
            return self._create_examples(self._read_json(path), "train")
        else:
            return self._create_examples(self._read_json(path), "train")[:self.k]

    def get_dev_examples(self, path=""):
        """See base class."""
        path += "/dev.json"
        logger.info("LOOKING AT {}".format(path))
        return self._create_examples(self._read_json(path), "dev")

    def get_test_examples(self, path=""):
        """See base class."""
        logger.info("LOOKING AT {}".format(path))
        path += "/test.json"
        return self._create_examples(self._read_json(path), "test")

    def get_labels(self):
        """See base class."""
        return ["motivation", "background", "uses",  "extends", "similarities",
                "differences",  "future_work"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            guid = line["id"]#"%s-%s" % (set_type, i)
            text = " ".join(line["x"]) if isinstance(line["x"], list) else line["x"]
            label = line["y"].split(" ")
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples


class OurClassificationProcessorJurgens(DataProcessor):
    """Processor for the classification version of our data set."""
    def __init__(self, k=None):
        super()
        self.k = k

    def _read_json(self, path):
        with open(path) as f_in:
            return json.load(f_in)

    def get_train_examples(self, path=""):
        """See base class."""
        path += "/train.json"
        logger.info("LOOKING AT {}".format(path))
        if self.k is None:
            return self._create_examples(self._read_json(path), "train")
        else:
            return self._create_examples(self._read_json(path), "train")[:self.k]

    def get_dev_examples(self, path=""):
        """See base class."""
        path += "/dev.json"
        logger.info("LOOKING AT {}".format(path))
        return self._create_examples(self._read_json(path), "dev")

    def get_test_examples(self, path=""):
        """See base class."""
        logger.info("LOOKING AT {}".format(path))
        path += "/test.json"
        return self._create_examples(self._read_json(path), "test")

    def get_labels(self):
        """See base class."""
        return ["Uses", "Motivation", "Future", "Extends", "CompareOrContrast", "Background"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        labels = {
            "motivation": "Motivation",
            "background": "Background",
            "uses": "Uses",
            "extends": "Extends",
            "similarities": "CompareOrContrast",
            "differences": "CompareOrContrast",
            "future_work": "Future"
        }
        for i, line in enumerate(lines):
            guid = line["id"]#"%s-%s" % (set_type, i)
            text = " ".join(line["x"]) if isinstance(line["x"], list) else line["x"]
            label = [labels[l] for l in line["y"].split(" ")] # we take all labels
            #label = labels[line["y"].split(" ")[0]] # we only take the first label here, not for training!
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples


class JurgensProcessor(DataProcessor):
    """Processor for the cohan data set."""
    def __init__(self, k=None):
        super()
        self.k = k

    def _read_jsonl(self, path):
        with jsonlines.open(path) as reader:
            return list(reader)

    def get_train_examples(self, path="./data/jurgens/"):
        """See base class."""
        path += "train.jsonl"
        logger.info("LOOKING AT {}".format(path))
        if self.k is None:
            return self._create_examples(self._read_jsonl(path), "train")
        else:
            return self._create_examples(self._read_jsonl(path), "train")[:self.k]

    def get_dev_examples(self, path="./data/jurgens/"):
        """See base class."""
        path += "dev.jsonl"
        logger.info("LOOKING AT {}".format(path))
        return self._create_examples(self._read_jsonl(path), "dev")

    def get_test_examples(self, path="./data/jurgens/"):
        """See base class."""
        path += "test.jsonl"
        logger.info("LOOKING AT {}".format(path))
        return self._create_examples(self._read_jsonl(path), "test")

    def get_labels(self):
        """See base class."""
        return ["Uses", "Motivation", "Future", "Extends","CompareOrContrast", "Background"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = line["text"]
            label = line["intent"]
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))

        return examples


class JurgensBinaryProcessor(DataProcessor):
    """Processor for the cohan data set."""
    def __init__(self, k=None, label=None):
        super()
        self.k = k
        self.label = label

    def _read_jsonl(self, path):
        with jsonlines.open(path) as reader:
            return list(reader)

    def get_train_examples(self, path="./data/jurgens/"):
        """See base class."""
        path += "train.jsonl"
        logger.info("LOOKING AT {}".format(path))
        if self.k is None:
            return self._create_examples(self._read_jsonl(path), "train")
        else:
            return self._create_examples(self._read_jsonl(path), "train")[:self.k]

    def get_dev_examples(self, path="./data/jurgens/"):
        """See base class."""
        path += "dev.jsonl"
        logger.info("LOOKING AT {}".format(path))
        return self._create_examples(self._read_jsonl(path), "dev")

    def get_test_examples(self, path="./data/jurgens/"):
        """See base class."""
        path += "test.jsonl"
        logger.info("LOOKING AT {}".format(path))
        return self._create_examples(self._read_jsonl(path), "test")

    def get_labels(self):
        """See base class."""
        return ["True", "False"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = line["text"]
            if line["intent"] == self.label:
                label = "True"
            else:
                label = "False"
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))

        return examples




tasks_num_labels = {
    "jurgens": 6,
    "ours": 7,
    "ours_jurgens": 6,
    "jurgens_binary": 2
}

citances_processors = {
    "ours": OurClassificationProcessor,
    "jurgens": JurgensProcessor,
    "jurgens_binary": JurgensBinaryProcessor,
    "ours_jurgens": OurClassificationProcessorJurgens
}

citances_output_modes = {
    "ours": "multilabel_classification",
    "ours_jurgens": "classification",
    "jurgens_binary": "classification",
    "jurgens": "classification",
}

