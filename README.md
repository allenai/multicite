# MultiCite

Modeling realistic citations requires moving beyond the single-sentence single-label setting.

## Dataset

The full dataset can be found at `data/full-v20210504.json`.  It has the following schema:

```json
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

## License

MultiCite is released under the CC BY-NC 2.0 as it is derived on top of [S2ORC](https://github.com/allenai/s2orc#license).  By using MultiCite, you are agreeing to its usage terms.

## Citation

If using this dataset, please cite:

```
TBD
```