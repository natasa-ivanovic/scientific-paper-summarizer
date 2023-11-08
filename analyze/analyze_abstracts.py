import os
import statistics

TRAIN_PATH = "./../train-dataset"

sentences_lengths = []
abstract_words = []
abstract_chars = []
sentences_dict = {}
for filename in os.listdir(TRAIN_PATH):
    number = filename.split(".")[0]
    base_path = os.path.join(TRAIN_PATH, filename) + "/"
    f = base_path + "abstract.txt"
    print("---\nprocessing", f)

    with open(f, encoding="utf-8") as file:
        text = file.read()
        text = text.replace("\n", "")
        abstract_chars.append(len(text))
        # print(text)
        if "ˇ" in text:
            print("WARNING - BROKEN ˇ !!!!!!!!!")
        if "´" in text:
            print("WARNING - BROKEN ´ !!!!!!!!!")
        # print(text)
        lines = text.split(".")
        sentences = [x for x in lines if x != '']
        if len(sentences) == 0:
            print("WARNING - NO ABSTRACT!!!!!!!!!")
        # print(sentences)
        # print(len(sentences))
        sentences_dict[f] = len(sentences)
        sentences_lengths.append(len(sentences))
        words = sum([len(sentence.split(" ")) for sentence in sentences])
        abstract_words.append(words)

for k, v in sentences_dict.items():
    print(k, ":", v)
print("Max num of sentences in abstracts:", max(sentences_lengths))
print("Min num of sentences in abstracts:", min(sentences_lengths))
print("Average num of sentences in abstracts:", statistics.mean(sentences_lengths))
print("Median num of sentences in abstracts:", statistics.median(sentences_lengths))
print("Low Median num of sentences in abstracts:", statistics.median_low(sentences_lengths))
print("High Median num of sentences in abstracts:", statistics.median_high(sentences_lengths))
print("---")
print("Max num of words:", max(abstract_words))
print("Min num of words:", min(abstract_words))
print("Average num of words:", statistics.mean(abstract_words))
print("Median num of words:", statistics.median(abstract_words))
print("Low Median num of words:", statistics.median_low(abstract_words))
print("High Median num of words:", statistics.median_high(abstract_words))
print("---")
print("Max num of characters:", max(abstract_chars))
print("Min num of characters:", min(abstract_chars))
print("Average num of characters:", statistics.mean(abstract_chars))
print("Median num of characters:", statistics.median(abstract_chars))



