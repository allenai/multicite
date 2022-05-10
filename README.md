# MultiCite

Modeling realistic citations requires moving beyond the single-sentence single-label setting.

See our preprint on [arXiv](https://arxiv.org/abs/2107.00414).

## Dataset

The full dataset can be found at `data/full-v20210918.json`.  It has the following schema:

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

## Intent IDs

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

## Models

All models trained are available on Huggingface:

1. Multi-label Citation Intent Classification
https://huggingface.co/allenai/multicite-multilabel-scibert
https://huggingface.co/allenai/multicite-multilabel-roberta-large

2. Citation Context Identification
tba

3. Paper-Level Citation Intent Q&A
tba

## License

MultiCite is released under the CC BY-NC 2.0 as it is derived on top of [S2ORC](https://github.com/allenai/s2orc#license).  By using MultiCite, you are agreeing to its usage terms.

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
