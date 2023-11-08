import os
import fitz
import re
from srtools import cyrillic_to_latin

TEST_PATH = "test-dataset"
TRAIN_PATH = "train-dataset"
FAILED_PATH = "failed"

CURRENT_PATH = TEST_PATH


def save_content(path, content):
    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)


def get_the_latest_occurrence(substring, string):
    idx_list = [m.start() for m in re.finditer(substring.lower(), string.lower())]
    if len(idx_list) != 0:
        return max(idx_list)
    return -1


def get_occurrence(substring, string, referent_idx):
    idx_list = [m.start() for m in re.finditer(substring.lower(), string.lower())]
    if len(idx_list) == 0:
        return -1
    for idx in idx_list:
        if idx > referent_idx:
            return idx
    return -1


def remove_page_numbers(page_elements):
    try:
        int(page_elements[-1])
        del page_elements[-1]
    except Exception:
        pass


def get_idx_pattern(page, word_1, word_2):
    pattern = rf"{re.escape(word_1)}\s*{re.escape(word_2)}"
    match = re.search(pattern, page, flags=re.IGNORECASE)
    if match:
        return match.group(), match.start()

    # print("NO", word_1, word_2, "!")
    return word_1 + word_2, -1


def get_keywords_idx(page, dictionary):
    key_en, keywords_idx_en = get_idx_pattern(page, "key", "words")
    key_srb, keywords_idx_srb = get_idx_pattern(page, "ključne", "reči")
    if keywords_idx_srb == -1:
        key_srb, keywords_idx_srb = get_idx_pattern(page, "ključne", "riječi")

    dictionary[key_en] = keywords_idx_en
    dictionary[key_srb] = keywords_idx_srb
    if keywords_idx_en > keywords_idx_srb:
        return key_srb, key_en, keywords_idx_en
    return key_srb, key_en, keywords_idx_srb


# switch to regex later
def get_intro_idx(page, keywords_idx):
    intro_idx = page.lower().find("1. uvod")
    if intro_idx == -1:
        intro_idx = page.lower().find("1.\nuvod")
    if intro_idx == -1:
        intro_idx = page.lower().find("1 uvod")
    if intro_idx == -1:
        intro_idx = page.lower().find("1\nuvod")
    if intro_idx == -1:
        intro_idx = page.lower().find("uvod")
        if intro_idx < keywords_idx:
            intro_idx = -1
    if intro_idx == -1:
        intro_idx = get_occurrence("1", page, keywords_idx)
    if intro_idx == -1:
        print("ERROR - NO INTRODUCTION!")
    return intro_idx


def build_dict_abstract_srb(page, dictionary):
    key, idx = get_idx_pattern(page, "kratak", "sadržaj")
    if idx == -1:
        key = "Kratak\nsadržaj"
        idx = page.find(key)
    if idx == -1:
        key = "Sažetak"
        idx = page.find(key)
    # if idx == -1:
    #     print("No serbian abstract!")

    dictionary[key] = idx
    return key


def build_dict_abstract_eng(abstract, dictionary):
    key = "Abstract"
    idx = abstract.find(key)
    if idx == -1:
        key = "Abrstact"
        idx = abstract.find(key)
    # if idx == -1:
    #     print("No english abstract!")

    dictionary[key] = idx
    return key


def collect_doc_pages(document):
    pages = []
    for page in document:
        page_text = page.get_text()
        page_text = cyrillic_to_latin(page_text)
        page_text_elements = page_text.strip().split("\n")
        # remove page numbers if exist
        remove_page_numbers(page_text_elements)
        page_text = "\n".join(page_text_elements)
        pages.append(page_text)
    return pages


def process_first_page(pages):
    first_page = pages[0]
    first_page_lines = first_page.strip().split("\n")
    # remove udk and doi lines
    del first_page_lines[1]
    del first_page_lines[1]
    first_page = "\n".join(first_page_lines)
    return first_page


def get_end_idx(dictionary, start_idx):
    end_abs_idx = -1
    # find the index after abs_idx in dictionary
    for key in sorted(dictionary, key=dictionary.get):
        if dictionary[key] > start_idx:
            # print(">")
            # print("END PHRASE:", key)
            # print("END INDEX:", end_abs_idx)
            # print(">")
            return dictionary[key]
    return -1


def process_document():
    for filename in os.listdir(CURRENT_PATH):
        number = filename.split(".")[0]
        base_path = os.path.join(CURRENT_PATH, filename) + "/"
        f = base_path + number + ".pdf"
        print("---\nprocessing", f)

        with fitz.open(f) as doc:
            pages = collect_doc_pages(doc)

        abstract_elements = {}
        first_page = process_first_page(pages)
        start_idx = first_page.lower().find("zbornik")
        key_srb, _, keywords_idx = get_keywords_idx(first_page, abstract_elements)
        introduction_idx = get_intro_idx(first_page, keywords_idx)
        extracted_abstract = first_page[start_idx:introduction_idx]
        build_dict_abstract_eng(extracted_abstract, abstract_elements)
        abs_key_srb = build_dict_abstract_srb(extracted_abstract, abstract_elements)
        print(abstract_elements)

        # get the index of "kratak sadrzaj"
        abs_idx = abstract_elements[abs_key_srb]
        if abs_idx == -1:
            print("ERROR - NO ABSTRACT!")
        end_abs_idx = get_end_idx(abstract_elements, abs_idx)
        final_abstract = extracted_abstract[abs_idx+len(abs_key_srb)+2:end_abs_idx].strip()
        print(">")
        print(final_abstract)

        # get the index of "ključne reči"
        keywords_srb_idx = abstract_elements[key_srb]
        end_keywords_idx = get_end_idx(abstract_elements, keywords_srb_idx)
        if end_keywords_idx == -1:
            print("ERROR - NO KEYWORDS!")
        if end_keywords_idx != -1:
            final_keywords = extracted_abstract[keywords_srb_idx + len(key_srb)+1:end_keywords_idx].strip()
        else:
            # end of page
            final_keywords = extracted_abstract[keywords_srb_idx + len(key_srb)+1:].strip()
        print(">")
        print(final_keywords)

        first_page_updated = first_page.replace(extracted_abstract, "")
        pages[0] = first_page_updated
        text = "\n".join(pages)
        save_content(base_path + "abstract.txt", final_abstract)
        save_content(base_path + "keywords.txt", final_keywords)
        save_content(base_path + number + ".txt", text)
        break


if __name__ == '__main__':
    process_document()



