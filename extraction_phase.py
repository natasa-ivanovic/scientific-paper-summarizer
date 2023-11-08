import math
import os
import re
import time

import networkx as nx
import numpy as np
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# remove dots after numbers
REGEX_PATTERN = re.compile("(?<=\d)(\.)(?!\d)")


def read_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_file_lines(path):
    with open(path, encoding="utf-8") as f:
        return f.readlines()


STOP_WORDS = read_file_lines("stop_words.txt")


def write_file(path, content, use_newline=False):
    with open(path, 'w', encoding="utf-8") as f:
        if use_newline:
            for c in content:
                f.write(c + '\n')
        else:
            f.write(content)


def tokenize_sentences(text):
    return sent_tokenize(text)


def clean_chapter(chapter):
    chapter_title, chapter_text = chapter
    # remove special char (any that are not in the specified set [] in regex)
    chapter_text_cleaned = re.sub(
        "[^a-zA-Z0-9ščćžđŠČĆŽĐ \n;,:()-.!]", '', chapter_text)
    # remove dots after numbers
    number_pattern = r'(\d+)\.'
    chapter_text_cleaned = re.sub(number_pattern, r'\1', chapter_text_cleaned)
    # remove abbreviations
    with open('abbreviations.txt', "r", encoding='utf-8') as file:
        abbreviations = [line.strip() for line in file]
    for abbreviation in abbreviations:
        if abbreviation in chapter_text_cleaned:
            non_dot_abr = abbreviation.split('.')[0]
            chapter_text_cleaned = chapter_text_cleaned.replace(
                abbreviation, non_dot_abr)
    return chapter_title, chapter_text_cleaned


def process_paper(paper_path, title_path):
    paper = read_file(paper_path)
    paper = remove_literature_and_biography(paper)
    paper = remove_references(paper)
    paper = remove_note(paper)
    chapters = split_paper(title_path, paper)
    # for each chapter, remove some more special characters
    chapters = [clean_chapter(chapter) for chapter in chapters]
    return chapters


def split_paper(title_path, paper):
    titles = read_file_lines(title_path)
    # print(titles)
    paper_list = paper.split("\n")
    paper = ""

    break_line = False
    for word in paper_list:
        if break_line:
            paper += word
        else:
            paper += " " + word
        if word.endswith("-"):
            break_line = True
        else:
            break_line = False
    paper = paper.strip()
    paper = re.sub(" +", " ", paper)
    pair_indices = []
    for title in titles:
        clean_title = title.replace("\n", "")
        clean_title = re.sub(" +", " ", clean_title)
        # remove references from titles
        clean_title = re.sub(r'\s*\[\d+]', '', clean_title)
        # print(clean_title)
        try:
            idx = paper.index(clean_title)
        except ValueError:
            # exclude BIOGRAFIJA/ZAHVALNOST - outliers
            if "biografija" or "zahvalnost" in title.lower():
                continue
            # remove number from the beginning
            clean_title = re.sub(r'^\d+\.?\s*', '', clean_title)
            # print(clean_title)
            idx = paper.index(clean_title)
        pair = (idx, len(clean_title))
        pair_indices.append(pair)

    return split_text_into_chapters(paper, pair_indices)


def split_text_into_chapters(text, titles_info):
    chapters = []
    for i, (start_idx, title_len) in enumerate(titles_info):
        title = text[start_idx:start_idx + title_len].strip()
        if i < len(titles_info) - 1:
            next_start_idx = titles_info[i + 1][0]
            chapter_text = text[start_idx + title_len:next_start_idx].strip()
        else:
            chapter_text = text[start_idx + title_len:].strip()
        chapters.append((title, chapter_text))
    return chapters


def remove_literature_and_biography(paper_text):
    lines = paper_text.strip().split("\n")

    # Find the line containing "literatura" as a section title
    literature_idx = None
    for idx, line in enumerate(lines):
        if re.match(r"\d+\.\s*literatura", line, re.IGNORECASE):
            literature_idx = idx
            break

    # If "literatura" section title is found, keep lines before it
    if literature_idx is not None:
        lines = lines[:literature_idx]

    # Join the lines back into the paper text
    paper_text = "\n".join(lines)

    return paper_text


def remove_references(paper_text):
    pattern1 = re.compile("\\[\\d]")
    pattern2 = re.compile("\\([0-9]+\\)")
    result = re.sub(pattern1, "", paper_text)
    result = re.sub(pattern2, "", result)
    return result


def remove_note(paper_text):
    paper_list = paper_text.split("\n")
    # find NAPOMENA
    # remove elem -1, elem, elem + 1, elem + 2
    # after del -> 4 times idx -1
    idx = next((i for i, v in enumerate(paper_list) if "NAPOMENA" in v), None)
    if idx is None:
        return "\n".join(paper_list)
    # part: ------...
    del paper_list[idx - 1]
    # part: NAPOMENA
    del paper_list[idx - 1]
    # part: half of the sentence
    del paper_list[idx - 1]
    # part: second half of the sentence
    del paper_list[idx - 1]
    return "\n".join(paper_list)


# Create vectors and calculate cosine similarity b/w two sentences
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w not in stopwords:
            vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w not in stopwords:
            vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)


# Create similarity matrix among all sentences
def build_similarity_matrix(sentences, stop_words):
    # create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(idx1, len(sentences)):
            if idx1 != idx2:
                res = sentence_similarity(
                    sentences[idx1], sentences[idx2], stop_words)
                similarity_matrix[idx1][idx2] = res
                similarity_matrix[idx2][idx1] = res

    return similarity_matrix


def extract_sentences(chapter):
    sentences = tokenize_sentences(chapter)
    sentence_similarity_matrix = build_similarity_matrix(sentences, STOP_WORDS)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentences = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    return [sentence_tuple[1] for sentence_tuple in ranked_sentences]


def get_top_ranked_sentences(ranked_sentences, number_of_sentences):
    extracted_sentences = []
    sentence_difference = number_of_sentences - len(ranked_sentences)
    if sentence_difference > 0:
        # if we don't get enough ranked sentences, get as many as we can and return the difference
        number_of_sentences = len(ranked_sentences)
    for i in range(number_of_sentences):
        extracted_sentences.append(ranked_sentences[i])

    return extracted_sentences, sentence_difference


# Methods checks if our distribution of sentences per chapter is valid
# (checking if there are enough available sentences for each chapter)
# sentences_per_chapter - format [UVOD_SENTENCES, ...MID SENTENCES, ZAKLJUCAK SENTENCES]
# chapters - array chapter recenica [(UVOD, "recenice"), (MID CHAPTER, "recenice")..., ("ZAKLJUCAK", "recenice")]
def is_chapter_split_successful(sentences_per_chapter, chapters):
    chapter_max_len = [len(tokenize_sentences(c[1])) for c in chapters]
    # print(chapter_max_len)
    result_array = []
    for max_len_num, current_sentences_num in zip(chapter_max_len, sentences_per_chapter):
        # check if too many sentences are allocated for the chapter
        if current_sentences_num > max_len_num:
            # this will be distributed on other chapters
            result_array.append(current_sentences_num - max_len_num)
        else:
            result_array.append(0)
    sum_difference = sum(result_array)
    if sum_difference == 0:
        return True, []
    else:
        return False, result_array


def rebalance_chapter_split(initial_sentences_per_chapter, sentences_overflow_per_chapter, chapters, total_number_sentences):
    current_sentences_per_chapter = initial_sentences_per_chapter
    current_overflow_per_chapter = sentences_overflow_per_chapter
    chapter_max_len = [len(tokenize_sentences(c[1])) for c in chapters]
    # just in case, if remaining sentences cannot be distributed equally
    longest_chapter_index = chapter_max_len.index(max(chapter_max_len))
    while True:
        total_difference = sum(current_overflow_per_chapter)
        number_of_chapters = len(chapters)
        # number of chapters that don't have overflow
        total_candidates_chapters = len([f for f in current_overflow_per_chapter if f == 0])
        # how many sentences can be added to each non-overflown chapters' score
        additional_per_chapter = math.floor(total_difference / total_candidates_chapters)
        # if above not possible, how much to add to the longest chapter
        remaining = total_difference % total_candidates_chapters
        new_chapter_split = []
        # example: one sentence is supposed to be added to five potential chapters
        # total_candidates_chapters = 5, additional_per_chapter = 0, remaining = 1
        if additional_per_chapter == 0:
            for idx, el in enumerate(current_sentences_per_chapter):
                if current_overflow_per_chapter[idx] != 0:
                    # if there is overflow, take the max sentences from that chapter
                    new_chapter_split.append(el - current_overflow_per_chapter[idx])
                else:
                    # if this is chapter with most sentences, add remainder as well
                    value_to_add = 0
                    if idx == longest_chapter_index:
                        # if the largest, increase the number of sentences
                        value_to_add += remaining
                    new_chapter_split.append(el + value_to_add)
        # sentences can be equally distributed
        else:
            # populate with chapters we want to exclude
            exlude_chapters = []
            for idx, (sentences, overflow, max_sentences) in enumerate(zip(current_sentences_per_chapter,
                                                                           current_overflow_per_chapter,
                                                                           chapter_max_len)):
                if overflow == 0 and max_sentences < (sentences + additional_per_chapter):
                    exlude_chapters.append(idx)
                    remaining += additional_per_chapter

            for idx, el in enumerate(current_sentences_per_chapter):
                if idx in exlude_chapters:
                    new_chapter_split.append(el)
                    continue
                if current_overflow_per_chapter[idx] != 0:
                    new_chapter_split.append(
                        el - current_overflow_per_chapter[idx])
                else:
                    # add to every chapter
                    value_to_add = additional_per_chapter
                    if idx == longest_chapter_index:
                        # if this is chapter with most sentences, add remainder as well
                        value_to_add += remaining
                    new_chapter_split.append(el + value_to_add)
            # print(new_chapter_split)
        is_success, errors = is_chapter_split_successful(
            new_chapter_split, chapters)
        if sum(new_chapter_split) == total_number_sentences and is_success:
            return new_chapter_split
        else:
            current_sentences_per_chapter = new_chapter_split
            current_overflow_per_chapter = errors


def process_chapters(chapters, total_number_sentences, extracted_sentences_per_chapter):
    # w_in_between = Total_Sum_Of_Weights / (n + 3)
    #
    # With w_in_between determined, you can then calculate w_first and w_final:
    #
    # w_first = 2 * w_in_between
    # w_final = 2 * w_in_between

    # must be even
    number_of_chapters = len(chapters)
    if number_of_chapters == 1:
        # use all sentences on the one chapter
        chapter_weight = total_number_sentences
        chapter_sentences, _sentence_difference = get_top_ranked_sentences([0], chapter_weight)
        return chapter_sentences
    elif number_of_chapters == 2:
        # if we have 2 chapters, split the sentences
        shared_weight = math.floor(total_number_sentences / 2)
        shared_weight_remaining = total_number_sentences % 2
        all_sentences = []
        sentences_per_chapter = [shared_weight +
                                 shared_weight_remaining, shared_weight]
        is_success, errors = is_chapter_split_successful(
            sentences_per_chapter, chapters)
        if not is_success:
            sentences_per_chapter = rebalance_chapter_split(
                sentences_per_chapter, errors, chapters, total_number_sentences)
        for ranked_sentences, sentences_to_extract in zip(extracted_sentences_per_chapter, sentences_per_chapter):
            chapter_sentences, _sentence_difference = get_top_ranked_sentences(
                ranked_sentences, sentences_to_extract)
            for x in chapter_sentences:
                all_sentences.append(x)
        return all_sentences
    else:
        sentences_in_between_weight = total_number_sentences / (number_of_chapters + 2)
        sentences_in_between = round(sentences_in_between_weight)
        sentences_start_end = round(2 * sentences_in_between_weight)

        total = 2 * sentences_start_end + sentences_in_between * (number_of_chapters - 2)
        difference = total_number_sentences - total

        sentences_first = sentences_start_end
        sentences_end = sentences_start_end

        # introduction, chapters in between, conclusion
        sentences_per_chapter = [sentences_first, *[sentences_in_between for _c in range(number_of_chapters - 2)],
                                 sentences_end]

        # handle case where we don't have exact match due to rounding
        if difference > 0:
            if difference % 2 == 1:
                # if difference is odd, take (difference/2) + 1 for start and difference/2 for end
                add_first = math.floor(difference / 2) + 1
                sentences_per_chapter[0] += add_first
                add_end = math.floor(difference / 2)
                sentences_per_chapter[number_of_chapters - 1] += add_end
            else:
                add_both = int(difference / 2)
                sentences_per_chapter[0] += add_both
                sentences_per_chapter[number_of_chapters - 1] += add_both
        # if difference is less than 0, remove from middle elements
        elif difference < 0:
            first_remove_index = 1
            for i in range(abs(difference)):
                while True:
                    if sentences_per_chapter[first_remove_index] > 0:
                        sentences_per_chapter[first_remove_index] -= 1
                        if first_remove_index == len(sentences_per_chapter):
                            first_remove_index == 0
                        else:
                            first_remove_index += 1
                        break

                    if first_remove_index == len(sentences_per_chapter):
                        first_remove_index == 0
                    else:
                        first_remove_index += 1

        is_success, errors = is_chapter_split_successful(
            sentences_per_chapter, chapters)
        if not is_success:
            sentences_per_chapter = rebalance_chapter_split(sentences_per_chapter, errors, chapters, total_number_sentences)
        all_sentences = []
        for ranked_sentences, sentences_to_extract in zip(extracted_sentences_per_chapter, sentences_per_chapter):
            chapter_sentences, sentence_difference = get_top_ranked_sentences(ranked_sentences, sentences_to_extract)
            for x in chapter_sentences:
                all_sentences.append(x)

        return all_sentences


if __name__ == '__main__':
    COMBINED_PATH = 'combined-dataset/'
    TRAIN_PATH = 'train-dataset/'
    TEST_PATH = 'test-dataset/'
    beginning = time.time()
    ROOT_PATH = COMBINED_PATH
    # folders = ["0", "1", "2", "3", "4", "5"]
    # folders = [str(x) for x in range(0, 100)]
    total_number_sentencess = [5, 10, 15, 20, 25]
    # total_number_sentencess = [22]
    # for folder in problem_folders:
    for folder in tqdm(os.listdir(ROOT_PATH)):
        base_path = ROOT_PATH + folder + "/"
        paper_path = base_path + folder + ".txt"
        title_path = base_path + folder + "-titles.txt"
        processed_paper_path = base_path + folder + "-processed.txt"
        chapters_path = base_path + folder + "-chapters.txt"
        # print(processed_paper_path, " STARTED")
        chapters = process_paper(paper_path, title_path)
        extracted_sentences_per_chapter = [
            extract_sentences(chapter[1]) for chapter in chapters]
        for total_number_sentences in total_number_sentencess:
            try:
                new_file_path = ROOT_PATH + folder + '/' + folder + '-' + str(
                    total_number_sentences) + '-sentences-extracted.txt'
                # if os.path.exists(new_file_path):
                #     continue
                summary_sentences = process_chapters(
                    chapters, total_number_sentences, extracted_sentences_per_chapter)
                # print(summary_sentences)
                write_file(new_file_path, summary_sentences, True)
            except Exception as e:
                print(str(folder))
                print(e)
