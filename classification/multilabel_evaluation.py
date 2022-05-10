"""
Evaluation code for multi-label classification evaluation
"""

from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def evaluate_strict(preds, golds, length=None, lengths=None):
    """
    Rewards if all labels match
    :param preds:
    :param golds:
    :return:
    """
    trues = 0
    falses = 0
    for i, pred in enumerate(preds):
        if lengths is not None and length is not None and lengths[i] == length:
            if set(pred) == set(golds[i]):
                trues += 1
            else:
                falses += 1
        elif length is None and lengths is None:
            if set(pred) == set(golds[i]):
                trues += 1
            else:
                falses += 1
    return trues/(trues + falses)


def evaluate_weak(preds, golds, length=None, lengths=None):
    """
    Rewards if any label matches
    :param preds:
    :param golds:
    :return:
    """
    trues = 0
    falses = 0
    for i, pred in enumerate(preds):
        if lengths is not None and length is not None and lengths[i] == length:
            if set(pred).intersection(set(golds[i])) != set():
                trues += 1
            else:
                falses +=1
        elif length is None and lengths is None:
            if set(pred).intersection(set(golds[i])) != set():
                trues += 1
            else:
                falses +=1
    return trues/(trues + falses)


def evaluate_per_class(preds, golds):
    """
    Evaluate each class separately, assumes that all classes appear in golds
    :param preds:
    :param golds:
    :return:
    """
    result = {}
    labels = set([item for sublist in golds for item in sublist])
    for l in labels:
        ps = [1 if l in pred else 0 for pred in preds]
        gs = [1 if l in gold else 0 for gold in golds]
        result[l] = precision_recall_fscore_support(gs, ps, average="binary")
    return result


def evaluate(preds, golds, label_list=None, output_path=None):
    """
    Given the predictions and golds, run the evaluation in several modes: weak, strict, ...
    :param preds:
    :param golds:
    :return:
    """
    all_results = {}
    if label_list is not None:
        preds = [[label_list[i] for i in l] for l in preds]
        golds = [[label_list[i] for i in l] for l in golds]
    result_strict = evaluate_strict(preds, golds)
    result_weak = evaluate_weak(preds, golds)
    result_per_class = evaluate_per_class(preds, golds)
    all_results["strict"] = result_strict
    all_results["weak"] = result_weak
    all_results["per_class"] = result_per_class
    print("Result strict: ")
    print(str(result_strict) + "\n")
    print("Result weak: ")
    print(str(result_weak) + "\n")
    print("Results per class: ")
    print(str(result_per_class))
    if output_path is not None:
        with open(output_path, "w") as f:
            f.write("Result strict: \n")
            f.write(str(result_strict) + "\n\n")
            f.write("Result weak: \n")
            f.write(str(result_weak) + "\n\n")
            f.write("Results per class: \n")
            f.write(str(result_per_class))
    return all_results

def create_predictions_from_probs(path, labels):
    all_predictions = []
    with open(path) as f:

        for i, line in enumerate(f.readlines()):
            if i % 2 == 0:
                l = ""
                l += line.strip() + " "
            else:
                l += line.strip()
                predictions = l.split("\t")[0]
                predictions = exec(predictions)
                probabilities = l.split("\t")[1]
                probabilities = exec(probabilities)
                if len(predictions) == 0:
                    predictions = [np.argmax(probabilities)]
                all_predictions.append([labels[p] for p in predictions])
    return all_predictions


def main():
    preds = []
    labels = {
        "motivation": "Motivation",
        "background": "Background",
        "uses": "Uses",
        "extends": "Extends",
        "similarities": "CompareOrContrast",
        "differences": "CompareOrContrast",
        "future_work": "Future",
        "": ""
    }
    new_preds = create_predictions_from_probs("./output/st-scibert_ours_10_context_3.0_2e-5_0_100/predictions_probs.txt",
                ["motivation", "background", "uses", "extends", "similarities", "differences",  "future_work"])
    for pred in new_preds:
        preds.append([labels[l] for l in pred])

    #with open("/net/nfs2.corp/s2-research/annel/citation_contexts/output/" +
    #          "st-scibert_ours_10_context_3.0_2e-5_0_100/predictions.txt") as f:
    #    for line in f.readlines():
    #        preds.append([labels[l] for l in line.strip().split(" ")])

    print(preds[0])
    print(len(preds))
    from classification import citances_processor
    proc = citances_processor.OurClassificationProcessorJurgens()
    gold = proc.get_test_examples("/./data/classification_10_context")
    gold = [e.label for e in gold]
    print(gold[0])
    print(len(gold))
    evaluate(preds, gold)

if __name__ == "__main__":
    main()