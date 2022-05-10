from classification.multilabel_evaluation import evaluate as evaluate_multilabel, evaluate_weak, evaluate_strict
from classification.citances_processor import citances_processors as processors
import json

processor = processors["ours"]()
test_examples = processor.get_test_examples("./data/classification_gold_context/")
train_examples = processor.get_test_examples("./data/classification_gold_context/")
train_labels = [e.label for e in train_examples]





def _read_json(path):
    with open(path) as f_in:
        return json.load(f_in)


test_data = _read_json("../data/classification_gold_context/test.json")
all_data = _read_json("../data/annotations_classification_labels_9_context_all_files.json")


def get_performance_for_length(preds, golds, lengths):
    all_scores = []
    for length in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        if length in lengths:
            support = sum([1 for l in lengths if l == length])
            all_scores.append(support)
            all_scores.append(evaluate_weak(preds, golds, length=length, lengths=lengths))
            all_scores.append(evaluate_strict(preds, golds, length=length, lengths=lengths))
    # else:
    #    print("No test instance for length %d" % length)
    all_scores.append(len(preds))
    all_scores.append(evaluate_weak(preds, golds))
    all_scores.append(evaluate_strict(preds, golds))
    return all_scores

lengths = []
golds = []
for d in test_data:
    for ad in all_data:
        if ad["id"] == d["id"]:
            lengths.append(ad["gold_length"])
            golds.append(d["y"].split(" "))

golds = [e.label for e in test_examples]
#predictions = [[random.choice(e.label)] for e in test_examples]
#predictions = [["background"] for e in test_examples]

#label_map = {
#    "Background": ["background"],
#    "Motivation": ["motivation"],
#    "CompareOrContrast": ["similarities", "differences"],
#    "Uses": ["uses"],
#    "Extends": ["extends"],
#    "Future": ["future_work"],
#}

label_map = {
    "background": "Background",
    "motivation": "Motivation",
    "similarities": "CompareOrContrast",
    "differences": "CompareOrContrast",
    "uses": "Uses",
    "extends": "Extends",
    "future_work": "Future",
}
# clean acl-acr predictions
#predictions = [label_map[line.strip()] for line in open("./data/st-scibert_jurgens_ours_jurgens_gold_context_4.0_2e-5_0_100/predictions.txt").readlines()]
predictions = [[line.strip()] for line in open(
    "../data/st-scibert_jurgens_ours_jurgens_gold_context_4.0_2e-5_0_100/predictions.txt").readlines()]
for i,gold in enumerate(golds):
    golds[i] = [label_map[g] for g in gold]

evaluate_multilabel(preds=predictions, golds=golds)
print(get_performance_for_length(predictions, golds, lengths))
