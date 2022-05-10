import json
from itertools import groupby


def convert_predictions_to_eval_script_input(path, gold_path):
    """
    Converts the predictions from the Qasper model to the Qasper eval script. The golds are needed as they contain the
    text and the predictions are only a list of zeros and ones
    :param path: path to the predictions
    :param gold_path: path to our gold test set, which contains the actual sentences
    :return:
    """
    predictions = []
    with open(path, errors="ignore") as f:
        for line in f.readlines():
            if "prediction:  " in line:
                predictions.append(json.loads(line.split("prediction:  ")[1]))
    gold_evidences = {}
    with open(gold_path, errors="ignore") as f:
        gold_data = json.load(f)
        for g,v in gold_data.items():
            for qa in v["qas"]:
                gold_evidences[qa["question_id"]] = qa["answers"][0]
                gold_evidences[qa["question_id"]]["full_text"] = v["full_text"][0]["paragraphs"]

    for i, p in enumerate(predictions):
        evidence = []
        gold = gold_evidences[p["question_id"]]
        # the led code always inserts an additional sentence classification token after the question
        p["predicted_evidence"] = p["predicted_evidence"][1:]
        for i,s in enumerate(gold["full_text"]):
            if i < len(p["predicted_evidence"]) and p["predicted_evidence"][i] == 1:
                evidence.append(s)
        p["predicted_evidence"] = evidence
        print(p)

    with open(path + ".eval", "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

def majority_sl(training_path, test_path):
    def get_intent(question):
        if "background" in question:
            return "bg"
        elif "future" in question:
            return "fw"
        elif "similarities" in question:
            return "si"
        elif "differences" in question:
            return "di"
        elif "motivation" in question:
            return "mo"
        elif "use" in question:
            return "us"
        elif "extend" in question:
            return "ex"
        else:
            print("something is wrong")

    # we know the majority answer, lets create it for every test
    predictions = []
    with open(test_path, errors="ignore") as f:
        test_data = json.load(f)
        for g, v in test_data.items():
            for qa in v["qas"]:
                p = {}
                p["question_id"] = qa["question_id"]
                p["predicted_answer"] = "No" if get_intent(qa["question"]) != "bg" else "Yes"
                p["predicted_evidence"] = []
                predictions.append(p)
    with open("../data/majority_sl.eval", "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")


def majority(training_path, test_path):
    gold_evidences = []
    majorities = {}
    counts = {}
    def get_intent(question):
        if "background" in question:
            return "bg"
        elif "future" in question:
            return "fw"
        elif "similarities" in question:
            return "si"
        elif "differences" in question:
            return "di"
        elif "motivation" in question:
            return "mo"
        elif "use" in question:
            return "us"
        elif "extend" in question:
            return "ex"
        else:
            print("something is wrong")

    def find_majority(k):
        myMap = {}
        maximum = ('', 0)  # (occurring element, occurrences)
        for n in k:
            if n in myMap:
                myMap[n] += 1
            else:
                myMap[n] = 1

            # Keep track of maximum on the go
            if myMap[n] > maximum[1]: maximum = (n, myMap[n])

        return maximum

    with open(training_path, errors="ignore") as f:
        training_data = json.load(f)
        for g,v in training_data.items():
            for qa in v["qas"]:
                if not get_intent(qa["question"]) in counts:
                    counts[get_intent(qa["question"])] = [qa["answers"][0]["answer"]["yes_no"]]
                else:
                    counts[get_intent(qa["question"])].append(qa["answers"][0]["answer"]["yes_no"])
        ans_dict = {
            True : "Yes",
            False: "No"
        }
        for intent, v in counts.items():
            majorities[intent] = ans_dict[find_majority(v)[0]]

        # we know the majority answer, lets create it for every test
        predictions = []
        with open(test_path, errors="ignore") as f:
            test_data = json.load(f)
            for g, v in test_data.items():
                for qa in v["qas"]:
                    p = {}
                    p["question_id"] = qa["question_id"]
                    p["predicted_answer"] = majorities[get_intent(qa["question"])]
                    p["predicted_evidence"] = []
                    predictions.append(p)
    with open("../data/majority.eval", "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")



def main():
    majority_sl("../data/qa_binary/ours_qa_train.json", "../data/qa_binary/ours_qa_test.json")

if __name__ == "__main__":
    main()