import random
import os
import shutil

NUM_OF_PAPERS = 2216
NUM_OF_TEST_DATASET = 221
PATH = "../combined-dataset/"
TEST_DATASET = "test-dataset/"
TRAIN_DATASET = "train-dataset/"

folders_to_copy = os.listdir(PATH)
test_folders = random.sample(folders_to_copy, NUM_OF_TEST_DATASET)
train_folders = list(set(folders_to_copy) - set(test_folders))

for folder in folders_to_copy:
    f = os.path.join(PATH, folder)
    if folder in test_folders:
        new_folder = TEST_DATASET + folder
        shutil.copytree(f, new_folder)
        print(f"Number {folder}: Moved to TESTING")
    elif folder in train_folders:
        new_folder = TRAIN_DATASET + folder
        shutil.copytree(f, new_folder)
        print(f"Number {folder}: Moved to TRAINING")
    else:
        print(f"Number {folder} something bad happened")
