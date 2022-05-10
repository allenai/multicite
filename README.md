# MultiCite (NAACL 2022)

Modeling realistic citations requires moving beyond the single-sentence single-label setting.

See our preprint on [arXiv](https://arxiv.org/abs/2107.00414).

## Dataset

### Full Dataset

The full dataset can be found at `data/full-v20210918.json`. It has the following schema:

```
{
  <paper_id>: {
    "x": [                  # contains full paper text split by sentences
      {
        "sent_id": <str>,   # globally unique identifier for this sentence
        "text": <str>       # citation mentions of target paper are wrapped in <span ..."></span> tags
      },
      {},
      {},
      ...
    ],
    "y": {
      <intent_id>: {
        "gold_contexts": [
            <List[str]>,    # list of sent_ids annotated as a context for this intent
            <List[str]>,    # different list of sent_ids annotated as a context for this intent
        ], 
        "cite_sentences": <List[str]>   # all sent_ids containing citation mention of target paper
      },
      <intent_id>: {},
      <intent_id>: {},
      ...
    }
  },
  <paper_id>: {},
  <paper_id>: {},
  ...
}
```

Intent IDs

```
{
    '@BACK@': 'Background',
    '@MOT@': 'Motivation',
    '@FUT@': 'Future Work',
    '@SIM@': 'Similar',
    '@DIF@': 'Difference',
    '@USE@': 'Uses',
    '@EXT@': 'Extention',
    '@UNSURE@': 'Unsure'
}
```

### Version for Classification Experiments

For the classification experiments described in the manuscript, we extracted the citations contexts from the full texts.
The versions we have used are provided in ```./data/classification_<context_size>_context```, where ```<context_size>```
reflects the size of the context extracted around the citation mention. For instance, in
```./data/classification_1_context```, the text to classify is always the sentence mentioning the cited work only,
while in ```./data/classification_gold_context``` the text to classify corresponds to the annotated gold context.

The structure is the following:

```
 [
     {
      "id": <instance_id>,
      "x": <text>,
      "y": <label>
     }
 ]
```

### Version for Q&A Experiments

Also for the Q&A experiments described in the manuscript, we needed to transform the data.
The versions we have used are provided in ```./data/classification_<context_size>_context```, where ```<context_size>```
reflects the size of the context extracted around the citation mention. For instance, in
```./data/classification_1_context```, the text to classify is always the sentence mentioning the cited work only,
while in ```./data/classification_gold_context``` the text to classify corresponds to the annotated gold context.

The code we used to convert the data is available in ```./qa/convert_ours_to_qa.py```.

The structure is the following:

```
[
    {
        <paper_id>: {
            "title": <paper_id>,
                "abstract": "",
                "full_text": [
                    {
                        "section_name": "",
                        "paragraphs": [
                           <full_text_as_list_of_sentences>
                        ]
                    }
               ],
               "qas": [
                        {
                           "question": "Does the paper cite <cited_work> for background information?",
                           "question_id": "ABC_8f0aab7fd30ffc56cc477b25e6bb16_00",
                           "answers": [
                              {
                                 "answer": {
                                    "unanswerable": false,
                                    "extracted_spans": [],
                                    "yes_no": <label-true/false>,
                                    "free_form_answer": "",
                                    "evidence": [
                                    <citation_context_if_label_true>
                                    ],
                                    "highlighted_evidence": []
                                 }
                              }
                           ]
                        }, 
                        {
                           "question": "Does the paper cite <cited_work> as motivation?",
                           "question_id": "ABC_8f0aab7fd30ffc56cc477b25e6bb16_02",
                           "answers": [
                              {
                                 "answer": {
                                    "unanswerable": false,
                                    "extracted_spans": [],
                                    "yes_no": <label-true/false>,
                                    "free_form_answer": "",
                                    "evidence": [
                                       <citation_context_if_label_true>
                                    ],
                                    "highlighted_evidence": []
                                 }
                              }
                           ]
                        }, ...
               ]
    }, ...
]
```

## Models

The model checkpoints trained are available on Huggingface:

1. Multi-label Citation Intent Classification

   https://huggingface.co/allenai/multicite-multilabel-scibert
   https://huggingface.co/allenai/multicite-multilabel-roberta-large


2. Citation Context Identification

   tba


3. Paper-Level Citation Intent Q&A

   https://huggingface.co/allenai/multicite-qa-qasper

## Experiments

### Classification
The scripts needed for running the multi-label classification experiments can be found in ```./classification```.
An example call is provided in ```./classification/run_classify_multilabel.sh```.

### Paper-level Citation Intent Q&A
For running these experiments, we used the original code from Dasigi et al., 2021. A script converting our data set to the Qasper 
Q&A format is ```./qa/convert_ours_to_qa.py```. Baseline code is available in ```./qa/eval_qa.py```.

## License

MultiCite is released under the CC BY-NC 2.0 as it is derived on top
of [S2ORC](https://github.com/allenai/s2orc#license). By using MultiCite, you are agreeing to its usage terms.

## Citation

If using this dataset, please cite:

```
@misc{multicite-lauscher-2021,
    title={{M}ulti{C}ite: {M}odeling realistic citations requires moving beyond the single-sentence single-label setting},
    author={Anne Lauscher and Brandon Ko and Bailey Kuehl and Sophie Johnson and David Jurgens and Arman Cohan and Kyle Lo},
    year={2021},
    eprint={2107.00414},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
