import json
import os
import random
import statistics


def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read()
        lines = lines.replace("\n", " ")
        return lines


def read_file_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return lines


def write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def read_dataset(path, range):
    dataset = []
    for folder in os.listdir(path):
        f = os.path.join(path, folder)
        f_sentences = f"{f}/{folder}-{range}-sentences-extracted.txt"
        f_abstract = f"{f}/abstract.txt"
        sentences = read_file(f_sentences)
        abstract = read_file(f_abstract)
        pair = {"original": sentences, "summary": abstract}
        dataset.append(pair)

    return dataset


def read_dataset_v1(path, range):
    dataset = []
    words_input = []
    for folder in os.listdir(path):
        f = os.path.join(path, folder)
        f_sentences = f"{f}/{folder}-{range}-sentences-extracted.txt"
        f_abstract = f"{f}/abstract.txt"
        sentences = read_file(f_sentences)
        abstract = read_file(f_abstract)
        abstract = abstract.replace("-", "")
        abstract = abstract.replace("\n", " ")
        sentences = sentences.replace("\n", " ")
        sentences = sentences.replace("-", "")
        words_input.append(len(sentences.split(" ")))
        pair = {"original": sentences.lower(), "summary": abstract.lower()}
        dataset.append(pair)

    print(words_input)
    print("Max num of words:", max(words_input))
    print("Min num of words:", min(words_input))
    print("Average num of words:", statistics.mean(words_input))
    print("Median num of words:", statistics.median(words_input))
    return dataset


def read_dataset_original(path, range):
    originals = []
    for folder in os.listdir(path):
        f = os.path.join(path, folder)
        f_sentences = f"{f}/{folder}-{range}-sentences-extracted.txt"
        sentences = read_file(f_sentences)
        originals.append(sentences)
    return originals


def read_abstracts_and_titles():
    path_test = "helper/headlines/txt/headlines_test_extracted.txt"
    path_train = "helper/headlines/txt/headlines_train_extracted.txt"
    lines_test = read_file_lines(path_test)
    lines_train = read_file_lines(path_train)
    test_dataset = []
    middle_dataset = []
    all_elements = []
    # titles_list = []

    for line in lines_test:
        elements = line.split("_#$*$#_ ")
        num = elements[0]
        title = elements[1].strip()
        abstract_path = f"test-dataset/{num}/abstract.txt"
        abstract = read_file(abstract_path)
        dataset_json = {
            "original": abstract.lower(),
            "summary": title.lower()
        }
        test_dataset.append(dataset_json)
        all_elements.append(dataset_json)

    for line in lines_train:
        elements = line.split("_#$*$#_ ")
        num = elements[0]
        title = elements[1].strip()
        abstract_path = f"train-dataset/{num}/abstract.txt"
        abstract = read_file(abstract_path)
        dataset_json = {
            "original": abstract.lower(),
            "summary": title.lower()
        }
        middle_dataset.append(dataset_json)
        all_elements.append(dataset_json)

    validation_size = int(len(middle_dataset) * 0.2)
    random.shuffle(middle_dataset)
    train_dataset = middle_dataset[validation_size:]
    val_dataset = middle_dataset[:validation_size]

    # print("Max num of characters:", max(titles_list))
    # print("Min num of characters:", min(titles_list))
    # print("Average num of characters:", statistics.mean(titles_list))
    # print("Median num of characters:", statistics.median(titles_list))

    print(f"test: {len(test_dataset)}")
    print(f"train: {len(train_dataset)}")
    print(f"val: {len(val_dataset)}")

    test_dict = {"data": test_dataset}
    train_dict = {"data": train_dataset}
    val_dict = {"data": val_dataset}

    dataset = {
        "data": {
            "train": train_dict,
            "val": val_dict,
            "test": test_dict
        }
    }
    write_json('helper/headlines/headlines_dataset.json', dataset)
    write_json('helper/headlines/headlines_train_dataset.json', train_dict)
    write_json('helper/headlines/headlines_val_dataset.json', val_dict)
    write_json('helper/headlines/headlines_test_dataset.json', test_dict)

    # Writing to file
    with open('helper/headlines/dataset.jsonl', "w", encoding="utf-8") as file:
        for el in all_elements:
            file.write(f"{el}\n")


def store_extracted_dataset_and_abstracts(range):
    path1 = "train-dataset/"
    path2 = "test-dataset/"
    data = read_dataset_v1(path1, range)
    test_data = read_dataset_v1(path2, range)
    validation_size = int(len(data) * 0.2)
    train_dataset = data[validation_size:]
    val_dataset = data[:validation_size]
    print(f"train: {len(train_dataset)}")
    print(f"val: {len(val_dataset)}")
    print(f"test: {len(test_data)}")

    train_dict = {"data": train_dataset}
    val_dict = {"data": val_dataset}
    test_dict = {"data": test_data}

    dataset = {
        "data": {
            "train": train_dict,
            "val": val_dict,
            "test": test_dict
        }
    }

    # EXTRACT TRAIN - RANGE 5
    # Max num of words: 685
    # Min num of words: 73
    # Average num of words: 184.46740220661985
    # Median num of words: 171.0

    # EXTRACT TEST - RANGE 5
    # Max num of words: 519
    # Min num of words: 85
    # Average num of words: 192.18099547511312
    # Median num of words: 174

    # EXTRACT TRAIN - RANGE 10
    # Max num of words: 1009
    # Min num of words: 154
    # Average num of words: 343.19057171514544
    # Median num of words: 327.0

    # EXTRACT TEST - RANGE 10
    # Max num of characters: 795
    # Min num of characters: 172
    # Average num of characters: 355.4886877828054
    # Median num of characters: 338

    # EXTRACT TRAIN - RANGE 15
    # Max num of words: 1207
    # Min num of words: 249
    # Average num of words: 503.66349047141426
    # Median num of words: 485.0

    # EXTRACT TEST - RANGE 15
    # Max num of words: 1205
    # Min num of words: 255
    # Average num of words: 522.0316742081448
    # Median num of words: 500

    # EXTRACT TRAIN - RANGE 20
    # Max num of words: 1420
    # Min num of words: 314
    # Average num of words: 643.8716148445336
    # Median num of words: 622.0

    # EXTRACT TEST - RANGE 20
    # Max num of words: 1323
    # Min num of words: 351
    # Average num of words: 668.2036199095022
    # Median num of words: 651
    write_json(f'helper/papers/{range}/dataset_range_{range}.json', dataset)
    write_json(f'helper/papers/{range}/dataset_train_range_{range}.json', train_dict)
    write_json(f'helper/papers/{range}/dataset_val_range_{range}.json', val_dict)
    write_json(f'helper/papers/{range}/dataset_test_range_{range}.json', test_dict)


# store_extracted_dataset_and_abstracts(20)

