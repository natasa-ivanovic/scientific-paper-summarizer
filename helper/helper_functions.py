import re


def load_file_lines(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def remove_literature_and_biography(paper_text):
    literature_idx = paper_text.lower().find("literatura")
    paper_text = paper_text[0:literature_idx]
    paper_lines = paper_text.strip().split("\n")
    pattern = re.compile("[0-9]{1,2}[.]\\d?")
    match = pattern.match(paper_lines[-1])
    if match:
        paper_lines = paper_lines[0:-1]
    paper_text = "\n".join(paper_lines)
    return paper_text


def remove_note(paper_text):
    paper_list = paper_text.split("\n")
    # find NAPOMENA
    # remove elem -1, elem, elem + 1, elem + 2
    # after del -> 4 times idx -1
    idx = next(i for i, v in enumerate(paper_list) if "NAPOMENA" in v)
    # part: ------...
    del paper_list[idx - 1]
    # part: NAPOMENA
    del paper_list[idx - 1]
    # part: half of the sentence
    del paper_list[idx - 1]
    # part: second half of the sentence
    del paper_list[idx - 1]
    return "\n".join(paper_list)


def remove_references(paper_text):
    pattern1 = re.compile("\\[\\d]")
    pattern2 = re.compile("\\([0-9]+\\)")
    result = re.sub(pattern1, "", paper_text)
    result = re.sub(pattern2, "", result)
    return result


def remove_scientific_sentences(paper_text):
    paper_text = re.sub("•", "", paper_text)
    paper_sentences = paper_text.split(".")
    pattern = re.compile('[^a-zA-Z0-9ščćžđŠČĆŽĐ \n;,.:()%=-]')
    new_list = [item for item in paper_sentences if not pattern.search(item)]
    paper_text = ".".join(new_list)
    return paper_text
