import os

DELETE_FAILED = False


def count_non_empty_lines(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            non_empty_line_count = 0
            for line in file:
                if line.strip():  # Check if the stripped line is not empty
                    non_empty_line_count += 1
            return non_empty_line_count
    except FileNotFoundError:
        print("ERROR: File not found. " + str(file_path))
        return 0


if __name__ == '__main__':
    TRAIN_PATH = 'train-dataset/'
    TEST_PATH = 'test-dataset/'
    COMBINED_PATH = 'combined-dataset/'
    ROOT_PATH = COMBINED_PATH
    total_number_sentencess = [5, 10, 15, 20, 25]

    success = 0
    total = 0
    failed_array = {}
    for folder in os.listdir(ROOT_PATH):
        is_failed = False
        total += 1
        for total_number_sentences in total_number_sentencess:
            new_file_path = ROOT_PATH + folder + '/' + folder + '-' + str(
                total_number_sentences) + '-sentences-extracted.txt'
            if not os.path.exists(new_file_path):
                print("ERROR: doesn't exist", str(new_file_path))
                is_failed = True
                continue
            count = count_non_empty_lines(new_file_path)
            if count != total_number_sentences:
                new_array = failed_array[str(folder)] if str(folder) in failed_array != None else []
                new_array.append(total_number_sentences)
                failed_array[str(folder)] = new_array
                is_failed = True
                if DELETE_FAILED:
                    os.remove(new_file_path)
        if not is_failed:
            success += 1
            
    print('total - ', success)
    print('total success - ', success)
    print('total failed - ', len(failed_array.keys()))
    for key, value in failed_array.items():
        print(key + ' - ' + str(value))
            
