import json
import re

def get_qa_template():
    """
    :return:
    """
    return {"question": "",
            "question_id": "",
            "answers": []
    }

def get_section_template():
    """
    :return:
    """
    return {"section_name": "",
            "paragraphs": []
    }

def get_paper_template():
    """
    :return:
    """
    return {"title":"",
            "abstract": "",
            "full_text": [],
            "qas": []}

def get_answer_template():
    """
    free_form_answer should contain the mc options
    evidence should contain the gold context
    :return: 
    """
    return {"unanswerable": False,
              "extracted_spans": [],
              "yes_no": None,
              "free_form_answer": "",
              "evidence": [],
              "highlighted_evidence": []}


def load_data(path="./data/ours-2021-05-04-clean.json"):
    return json.load(open(path))

def load_context_file(path):
    with open(path) as f:
        return json.load(f)

def replace_markup(t):
    t = t.replace("<span style=\"background: yellow; display: inline-block\">", "<cite>")
    t = t.replace("</span>", "</cite>")
    return t

def extract_marker(annotation):
    sentences = [x["text"] for x in annotation["x"]]

    start = re.escape("<span style=\"background: yellow; display: inline-block\">")
    display_text = " ".join(sentences)
    parts = display_text.split("</span>")
    result = re.findall(start + '(.*)', parts[0])
    #result = [match.group(1) for match in result]
    if result:
        display_text = replace_markup(display_text)
        return result[0]
    else:
        return None

def extract_marker_from_gold_context(gold_context, annotation):
    start = re.escape("<span style=\"background: yellow; display: inline-block\">")
    text = " ".join([x["text"] for x in annotation["x"] if x["sent_id"] in gold_context])
    parts = text.split("</span>")
    result = re.findall(start + '(.*)', parts[0])
    if result:
        return result[0]
    return None

def get_question(marker):
    return "Why does the paper cite %s ? %s" % (marker, get_labels())


def get_binary_question(marker, intent):
    questions = {
        "ABC-N01": ("Does the paper cite %s for background information?" % marker),
        "ABC-N02": None,
        "ABC-N03": ("Does the paper cite %s as motivation?" % marker),
        "ABC-N04": ("Does the paper cite %s in relation to future work?" % marker),
        "ABC-N05": ("Does the paper express similarities towards %s?" % marker),
        "ABC-N06": ("Does the paper express differences towards %s?" % marker),
        "ABC-N07": ("Does the paper use something from %s?" % marker),
        "ABC-N08": ("Does the paper extend something from %s?" % marker),
        "ABC-N09": None
    }
    return questions[intent]

def get_labels():
    return "(A) motivation (B) background (C) uses (D) extends (E) similarities (F) differences (G) future_work"

def number_of_files(data):
    return len([key for key, value in data.items()])

def get_intents():
    return {
        "ABC-N01": "(B) background",
        "ABC-N02": None,
        "ABC-N03": "(A) motivation",
        "ABC-N04": "(G) future_work",
        "ABC-N05": "(E) similarities",
        "ABC-N06": "(F) differences",
        "ABC-N07": "(C) uses",
        "ABC-N08": "(D) extends",
        "ABC-N09": None
    }

def get_all_intents_for_context(context, annotation, intents):
    all_intents = []
    for i, intent_contexts in annotation["y"].items():
        intent = intents[i]
        if intent is not None:
            for b in intent_contexts["gold_contexts"]:
                if context == b:
                    all_intents.append(intent)
    # TODO: maybe we like to return it in the order in which they are mentioned in the question
    all_intents = set(all_intents)
    return list(all_intents)


def is_already_in_context_list(intents_context_pair, grouped_contexts):
    if intents_context_pair in grouped_contexts:
        return True
    return False


def group_contexts_by_overlapping_intents(annotation):
    """
    annotation contains all annotations per file
    :param annotation:
    :return:
    """
    intents = get_intents()
    grouped_annotations = []
    for i, intent_context in annotation["y"].items():
        # match overlapping labels
        intent_a = intents[i]
        if intent_a is not None:
            for gold_context in intent_context["gold_contexts"]:
                all_intents = get_all_intents_for_context(gold_context, annotation, intents)
                if not is_already_in_context_list((all_intents, gold_context), grouped_annotations):
                    grouped_annotations.append((all_intents, gold_context))
    return grouped_annotations

def create_qa_format(data, grouped=False, binary=True):
    papers = []
    intents = get_intents()
    for file, annotation in data.items():
        # filename = file.split("_")[1]
        section = get_section_template()
        # this is the fix
        section["paragraphs"] += [replace_markup(x["text"]) for x in sorted(annotation["x"], key=lambda k: k['sent_id'])]
        paper = get_paper_template()
        paper["full_text"] = [section]
        paper["title"] = file

        if not binary:
            marker = extract_marker(annotation)
            q = get_question(marker)
            qa = get_qa_template()
            qa["question"] = q
            qa["question_id"] = file

            if grouped:
                # old version in which we try to predict all gold contexts and also multilabel intents
                grouped_intents = group_contexts_by_overlapping_intents(annotation)
                for group in grouped_intents:
                    a = get_answer_template()
                    intents, context = group
                    a["free_form_answer"] = " ".join(intents)
                    a["evidence"] = [replace_markup(x["text"]) for x in annotation["x"] if x["sent_id"] in context]
                    qa["answers"].append({"answer": a})
            else:
                # new version: we just want to predict the intents with the gold context that appears first
                for key, intent_context in annotation["y"].items():
                    intent = intents[key]
                    if intent is not None:
                        a = get_answer_template()
                        a["free_form_answer"] = intent
                        # only the first gold context has to be extracted
                        context = intent_context["gold_contexts"][0]
                        a["evidence"] = [replace_markup(x["text"]) for x in annotation["x"] if x["sent_id"] in context]
                        qa["answers"].append({"answer": a})


            paper["qas"] = [qa]
        else:
            paper["qas"] = []
            all_intents = get_intents().keys()
            for i,int in enumerate(all_intents):
                if int in annotation["y"].keys():
                    intent_context = annotation["y"][int]
                    context = intent_context["gold_contexts"][0]
                    marker = extract_marker_from_gold_context(context, annotation)
                    q = get_binary_question(marker, int)
                    if q is not None:
                        qa = get_qa_template()
                        qa["question"] = q
                        qa["question_id"] = file + str(i)
                        # new version: we just want to predict the intents with the gold context that appears first
                        a = get_answer_template()
                        a["yes_no"] = True
                        # only the first gold context has to be extracted
                        a["evidence"] = [replace_markup(x["text"]) for x in annotation["x"] if x["sent_id"] in context]
                        qa["answers"].append({"answer": a})
                        paper["qas"].append(qa)
                else:
                    # create negative example
                    marker = extract_marker(annotation)
                    q = get_binary_question(marker, int)
                    if q is not None:
                        qa = get_qa_template()
                        qa["question"] = q
                        qa["question_id"] = file + str(i)
                        # new version: we just want to predict the intents with the gold context that appears first
                        a = get_answer_template()
                        a["yes_no"] = False
                        qa["answers"].append({"answer": a})
                        paper["qas"].append(qa)
        papers.append(paper)
    return papers

def train_dev_test_split_like(papers, path="./data/classification_1_context"):
    train_ids = set([example["id"].split("_")[0] for example in load_data(path + "/train.json")])
    dev_ids = set([example["id"].split("_")[0] for example in load_data(path + "/dev.json")])
    test_ids = set([example["id"].split("_")[0] for example in load_data(path + "/test.json")])
    train_papers = {}
    dev_papers = {}
    test_papers = {}
    for p in papers:
        if p["title"].split("_")[1] in train_ids:
            train_papers[p["title"]] = p
        elif p["title"].split("_")[1] in dev_ids:
            dev_papers[p["title"]] = p
        elif p["title"].split("_")[1] in test_ids:
            test_papers[p["title"]] = p
    #assert len(train_papers) == len(train_ids)
    #assert len(dev_papers) == len(dev_ids)
    #assert len(test_papers) == len(test_ids)
    print(str(sum([len(p["qas"][0]["answers"]) for k, p in train_papers.items()])))
    print(str(sum([len(p["qas"]) for k, p in train_papers.items()])))
    return train_papers, dev_papers, test_papers

def output(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=True)


def main():
    """
    In the file we load, there is for each paper an object;
    the inner part of the object key is the filename;
    :return:
    """
    data = load_data()
    print(str(len([key for key, value in data.items()])))
    print(str(len(set([key for key, value in data.items()]))))
    print(str(len(set([key.split("_")[1] for key, value in data.items()]))))
    papers = create_qa_format(data)
    train, dev, test = train_dev_test_split_like(papers)
    output(train, "../data/ours_led_qa/ours_qa_train.json")
    output(dev, "../data/ours_led_qa/ours_qa_dev.json")
    output(test, "../data/ours_led_qa/ours_qa_test.json")





if __name__ == '__main__':
    main()
