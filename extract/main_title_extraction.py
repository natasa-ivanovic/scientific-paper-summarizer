import re
import os
import fitz
from srtools import cyrillic_to_latin
from langdetect import detect
from tqdm import tqdm

TEST_PATH = "./../test-dataset"
TRAIN_PATH = ".././train-dataset"
FAILED_PATH = "failed"

CURRENT_PATH = TEST_PATH


def flags_decomposer(flags):
    """Make font flags human readable."""
    l = []
    if flags & 2 ** 0:
        l.append("superscript")
    if flags & 2 ** 1:
        l.append("italic")
    if flags & 2 ** 2:
        l.append("serifed")
    else:
        l.append("sans")
    if flags & 2 ** 3:
        l.append("monospaced")
    else:
        l.append("proportional")
    if flags & 2 ** 4:
        l.append("bold")
    return ", ".join(l)


def process_titles(path):
    # 3D ŠTAMPA DELA ZA NADOGRADNJU INDUSTRIJSKOG ROBOTA U OKVIRIMA INDUSTRIJE 4.0
    # 3D PRINTING OF AN INDUSTRIAL ROBOT UPGRADE PART IN INDUSTRY 4.0 FRAMEWORK
    with fitz.open(path) as doc:
        START_TITLE = False
        END_TITLE = False
        first_page = doc[0]
        blocks = first_page.get_text("dict", flags=11)["blocks"]
        title_parts = []
        for block in blocks:
            for line in block['lines']:
                spans = line["spans"]
                if len(spans) == 0:
                    continue
                for span in spans:
                    text = cyrillic_to_latin(span["text"])
                    if 'https://doi.org/' in text:
                        START_TITLE = True
                        break
                    if START_TITLE and len(title_parts) != 0 and (text.strip() == ''):
                        END_TITLE = True
                        break
                    if START_TITLE:
                        title_parts.append(text)
                if END_TITLE:
                    break
            if END_TITLE:
                return ' '.join([title.strip() for title in title_parts])
    return ''
file_path = "../helper/headlines/txt/headlines_test_extracted.txt"
subdirectories = [d for d in os.listdir(TEST_PATH) if os.path.isdir(os.path.join(TEST_PATH, d))]
sorted_subdirectories = sorted(subdirectories, key=lambda x: int(x))
with open(file_path, 'w', encoding='utf-8') as file:
    for folder in tqdm(sorted_subdirectories):
        f = os.path.join(TEST_PATH, folder)
        file_path = f"{f}/{folder}.pdf"
        title = process_titles(file_path)
        if title == '':
            title_to_write = 'ERROR IN EXTRACTION'
        else:
            title_to_write = title
        separator = "_#$*$#_"
        file.write(f"{folder}{separator}{title_to_write}\n")

# test = detect("3D ŠTAMPA DELA ZA NADOGRADNJU INDUSTRIJSKOG ROBOTA U OKVIRIMA INDUSTRIJE 4.0 ")
# print(test)