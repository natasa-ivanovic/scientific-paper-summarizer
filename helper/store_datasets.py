from sklearn.model_selection import train_test_split
import load_dataset as ld
import json

TEST_DATASET = "./../test-dataset/"
TRAIN_DATASET = "./../train-dataset/"
DATASET_RANGE = 10

test_data = ld.read_dataset(TEST_DATASET, DATASET_RANGE)
print(test_data)
data = ld.read_dataset(TRAIN_DATASET, DATASET_RANGE)
train_data_, val_data_ = train_test_split(data, test_size=0.2, shuffle=True)

test_dataset = {"data": test_data}
train_data = {"data": train_data_}
val_data = {"data": val_data_}
dataset = {
    "data": {
        "train ": train_data,
        "val": val_data,
        "test": test_dataset,
    }
}

with open(f'dataset_test_range_{DATASET_RANGE}.json', 'w', encoding='utf-8') as f:
    json.dump(test_dataset, f, ensure_ascii=False, indent=4)

with open(f'dataset_train_range_{DATASET_RANGE}.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open(f'dataset_val_range_{DATASET_RANGE}.json', 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=4)

with open(f'dataset_range_{DATASET_RANGE}.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

# with open("./headlines/headlines_dataset.json", encoding="utf-8") as f:
#     data = f.read().lower()
#     print(data)
#     with open('./headlines/headlines_dataset_lowercase.json', 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)