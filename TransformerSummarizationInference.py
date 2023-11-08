from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, pipeline
from transformers import TFAutoModelForSeq2SeqLM

# headlines
# model_checkpoint = "transformer-model-epochs-100"

RANGE = 5

# papers range 5
model_checkpoint = "./trained-models/transformer-model-paper-epochs-150-r5"
model = TFAutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_checkpoint)
summarizer = pipeline("summarization", model=model_checkpoint)

# headlines
# data_files = {
#     "train": "./helper/headlines/headlines_train_dataset.json",
#     "val": "./helper/headlines/headlines_val_dataset.json",
#     "test": "./helper/headlines/headlines_test_dataset.json"
# }


# papers
data_files = {
    "train": f"./helper/papers/{RANGE}/dataset_train_range_{RANGE}.json",
    "val": f"./helper/papers/{RANGE}/dataset_val_range_{RANGE}.json",
    "test": f"./helper/papers/{RANGE}/dataset_test_range_{RANGE}.json"
}

datasets = load_dataset("json", data_files=data_files, field="data")


def print_summary(dataset):
    # abstracts = dataset["original"]
    # titles = dataset["summary"]
    model_folder_path = f"./{model_checkpoint}"
    for idx in range(0, len(dataset)):
        if idx == 50:
            break
        output_file_path = f"{model_folder_path}/test-results.txt"
        output_file = open(output_file_path, "a", encoding="utf-8")
        input_text = dataset[idx]["original"]
        summarization_true = dataset[idx]["summary"]
        summarization_predicted = summarizer(input_text, max_length=200)[0].get("summary_text")
        in_text = f"INPUT\n{input_text}"
        pred_text = f"PREDICTION\n{summarization_predicted}"
        targ_text = f"TARGET\n{summarization_true}"
        print(in_text)
        print(pred_text)
        print(targ_text)
        output_file.write(f"{in_text}\n")
        output_file.write(f"{pred_text}\n")
        output_file.write(f"{targ_text}\n")
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(summarization_true, summarization_predicted)
        for key in scores:
            pair = f"{key}: {scores[key]}"
            print(pair)
            output_file.write(f"{pair}\n")
        output_file.write("--------------\n\n")
        print('--------------')
        # input()


train_dataset = datasets["train"]
val_dataset = datasets["val"]
test_dataset = datasets["test"]

print_summary(test_dataset)
