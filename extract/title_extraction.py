import re
import os
import fitz
from srtools import cyrillic_to_latin

TEST_PATH = "test-dataset"
TRAIN_PATH = "train-dataset"
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


def remove_garbage(lst):
    # remove '1', '1.', '123.', '12.   ', '12    ', ...
    pattern = r'^\d+(\.\s*)?\s*$'
    items = [item for item in lst if not re.match(pattern, item)]
    # remove _________________________ ...
    items = [item for item in items if not "___" in item]
    # remove if no letters
    items = ["" if not any(c.isalpha() for c in x) else x for x in items]
    items = [item for item in items if item.strip()]
    items = [item for item in items if "biografija" or "zahvalnost" in item.strip().lower()]
    return items


def is_introduction_present_in_block(block):
    uvod_span_found = False
    uvod_span_idx = -1
    uvod_line_idx = -1
    uvod_content = None
    for l in block["lines"]:
        for s in l["spans"]:
            text = cyrillic_to_latin(s["text"])
            font_name = s["font"]
            flags = flags_decomposer(s["flags"])
            if "UVOD" in text or "Uvod" in text and font_name == "Times New Roman,Bold" and "bold" in flags:
                uvod_span_found = True
                uvod_span_idx = l["spans"].index(s)
                uvod_content = s["text"]
                break
        if uvod_span_found:
            uvod_line_idx = block["lines"].index(l)
            break
    # remove all spans before UVOD in this block
    if uvod_span_found:
        if "1" not in uvod_content:
            uvod_span_idx = uvod_span_idx - 2
        block = remove_spans_before_introduction(block, uvod_line_idx, uvod_span_idx)
    return uvod_span_found, block


def filter_first_chapter(blocks):
    uvod_span_found = False
    uvod_span_idx = -1
    uvod_line_idx = -1
    block = None
    pattern = r'\b\d+\.\s(?![\d.])\S.*'
    for b in blocks:
        for l in b["lines"]:
            for s in l["spans"]:
                text = cyrillic_to_latin(s["text"])
                font_name = s["font"]
                flags = flags_decomposer(s["flags"])
                if bool(re.match(pattern, text)) and not is_subchapter(text) and font_name == "Times New Roman,Bold" and "bold" in flags:
                    uvod_span_found = True
                    uvod_span_idx = l["spans"].index(s)
                    break
            if uvod_span_found:
                uvod_line_idx = b["lines"].index(l)
                break
        if uvod_span_found:
            block = b
            break
    # remove all spans before first chapter in the marked block
    if uvod_span_found:
        block = remove_spans_before_introduction(block, uvod_line_idx, uvod_span_idx)
        block_idx = blocks.index(block)
        filtered_blocks = blocks[block_idx:]
        return filtered_blocks
    return blocks


def remove_spans_before_introduction(block, line_idx, span_idx):
    new_lines = block["lines"][line_idx:]
    new_spans = new_lines[0]["spans"][span_idx:]
    new_lines[0]["spans"] = new_spans
    block["lines"] = new_lines
    return block


def remove_spans_after_literature(block, line_idx, span_idx):
    new_lines = block["lines"][:line_idx + 1]
    new_spans = new_lines[0]["spans"][:span_idx + 1]
    new_lines[0]["spans"] = new_spans
    block["lines"] = new_lines
    return block


def is_subchapter(text):
    pattern = re.compile("\\d+\\.\\d+(?:\.\\d+)?")
    matches = pattern.match(text)
    return bool(matches)


def find_introduction(first_page):
    blocks = first_page.get_text("dict", flags=11)["blocks"]
    block = None
    uvod_found = False
    for b in blocks:
        uvod_found, block = is_introduction_present_in_block(b)
        if uvod_found:
            break
    if uvod_found:
        block_idx = blocks.index(block)
        filtered_blocks = blocks[block_idx:]
        return filtered_blocks
    else:
        filtered_blocks = filter_first_chapter(blocks)
        return filtered_blocks


def find_literature(last_page):
    blocks = last_page.get_text("dict", flags=11)["blocks"]
    literature_span_found = False
    literature_span_idx = -1
    literature_line_idx = -1
    block = None
    for b in blocks:
        for l in b["lines"]:
            for s in l["spans"]:
                text = cyrillic_to_latin(s["text"])
                font_name = s["font"]
                flags = flags_decomposer(s["flags"])
                literature_found = "LITERATURA" in text or "REFERENCE" in text or "BIBLIOGRAFIJA" in text
                if literature_found and font_name == "Times New Roman,Bold" and "bold" in flags:
                    literature_span_found = True
                    literature_span_idx = l["spans"].index(s)
                    break
            if literature_span_found:
                literature_line_idx = b["lines"].index(l)
                break
        if literature_span_found:
            block = b
            break

    # remove all spans after literature
    if literature_span_found:
        block = remove_spans_after_literature(block, literature_line_idx, literature_span_idx)
        block_idx = blocks.index(block)
        filtered_blocks = blocks[:block_idx + 1]
        # print(filtered_blocks)
        return filtered_blocks, literature_span_found
    return blocks, literature_span_found


def remove_literature(title_list):
    if len(title_list) >= 2:
        last_element = title_list[-1]
        literature_found = "LITERATURA" in last_element.upper() or "BIBLIOGRAFIJA" in last_element.upper() \
                           or "REFERENCE" in last_element.upper()
        if literature_found:
            del title_list[-1]
        last_element = title_list[-1].strip()
        elements = list(filter(None, last_element.split(".")))
        if len(elements) == 1 and elements[0].isdigit():
            del title_list[-1]


def is_title(text, title_parts, multiblock_title_parts):
    title_pattern = r'\b\d+\.(?:\s)?[^0-9].*'
    number_pattern = r'^\d+\.?$'
    contains_letters = any(char.isalpha() for char in text)
    if contains_letters and not text.isupper():
        return False
    if "NAPOMENA" in text or "LITERATURA" in text:
        return False
    if len(title_parts) >= 1 or len(multiblock_title_parts) >= 1:
        return not is_subchapter(text)
    else:
        return bool(re.match(number_pattern, text.strip())) or bool(re.match(title_pattern, text))


def process_titles():
    titles = {}
    for filename in os.listdir(CURRENT_PATH):
        number = filename.split(".")[0]
        base_path = os.path.join(CURRENT_PATH, filename) + "/"
        f = base_path + number + ".pdf"
        f_out = base_path + number + "-titles.txt"

        print("---\nprocessing", f)
        with fitz.open(f) as doc:
            # for page in doc:
            #     page_text = page.get_text()
            #     print(page_text)
            current_titles = []
            for page_num in range(doc.page_count):
                page = doc[page_num]
                blocks = page.get_text("dict", flags=11)["blocks"]

                is_multiblock_title_finished = False
                multiblock_title_parts = []
                # ako u lajnovima postoji vise od jednog elementa, znaci da nikada nece
                # biti multi block title
                is_multiblock_candidate = True
                for b in blocks:
                    # print(b)
                    # print("---")
                    lines = b["lines"]
                    title_parts = []
                    for l in lines:
                        spans = l["spans"]
                        if len(spans) == 0:
                            continue
                        if len(spans) > 1:
                            text = "".join([el["text"] for el in spans])
                            font_list = [el["font"] for el in spans if not el["text"].isspace()]
                            is_bold = all("Bold" in element for element in font_list)

                        else:
                            text = spans[0]["text"]
                            is_bold = "Bold" in spans[0]["font"]
                        text = cyrillic_to_latin(text)
                        if is_bold and is_title(text, title_parts, multiblock_title_parts):
                            if not is_multiblock_candidate and len(title_parts) >= 1:
                                new_title = "".join(title_parts)
                                if len(multiblock_title_parts) >= 1:
                                    new_title = "".join(multiblock_title_parts) + new_title
                                    multiblock_title_parts = []
                                current_titles.append(new_title)
                                title_parts = []
                            title_parts.append(text)
                            is_multiblock_candidate = True
                        else:
                            is_multiblock_candidate = False
                            if len(multiblock_title_parts) >= 1 and len(title_parts) == 0:
                                if "." in multiblock_title_parts[0]:
                                    current_titles.append("".join(multiblock_title_parts))
                                multiblock_title_parts = []
                                is_multiblock_candidate = True

                    if len(title_parts) != 0:
                        if is_multiblock_candidate:
                            multiblock_title_parts.append("".join(title_parts))
                            # is_multiblock_title_finished = False
                        else:
                            current_titles.append("".join(title_parts))
            # fallback for literature
            remove_literature(current_titles)
            current_titles = remove_garbage(current_titles)
            print(current_titles)
            if len(current_titles) == 0:
                print("NO TITLES!!!!!!!!!!!!!!!!!")
            write_titles(f_out, current_titles)
            titles[f] = current_titles
    return titles


def write_titles(file, titles):
    with open(file, 'a', encoding='utf-8') as file:
        for title in titles:
            file.write(title)
            file.write("\n")


if __name__ == '__main__':
    titles = process_titles()
    print("Writing in the file...")
    with open("titles.txt", 'a', encoding='utf-8') as file:
        for key in titles:
            file.write(key + "#&*" + "|".join(titles[key]))
            file.write("\n")

