import evaluate
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import TFAutoModelForSeq2SeqLM
from transformers import create_optimizer

EPOCHS = 20
BATCH_SIZE = 8

# model_checkpoint = "facebook/mbart-large-50"
model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Max num of characters: 1119
# Min num of characters: 112
# Average num of characters: 443.87330316742083
# Median num of characters: 411
max_input_length = 750
max_target_length = 160

data_files = {
    "train": "./helper/headlines/headlines_train_dataset.json",
    "val": "./helper/headlines/headlines_val_dataset.json",
    "test": "./helper/headlines/headlines_test_dataset.json"
}

datasets = load_dataset("json", data_files=data_files, field="data")


def preprocess_function(examples):
    model_inputs = tokenizer(
        # abstracts
        examples["original"],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        # titles
        examples["summary"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = datasets.map(preprocess_function, batched=True)
# print(tokenized_datasets)

rouge_score = evaluate.load("rouge")

# dynamically pad the inputs and outputs
# Since mT5 is an encoder-decoder Transformer model, one subtlety with preparing our batches is that during decoding
# we need to shift the labels to the right by one.
# This is required to ensure that the decoder only sees the previous ground truth labels
# and not the current or future ones, which would be easy for the model to memorize.
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")
features = [tokenized_datasets["train"][i] for i in range(2)]
# print(data_collator(features))

# print(tokenized_datasets["train"])
tf_train_dataset = model.prepare_tf_dataset(
    tokenized_datasets["train"],
    collate_fn=data_collator,  # padding, teacher forcing
    shuffle=True,
    batch_size=BATCH_SIZE
)

tf_eval_dataset = model.prepare_tf_dataset(
    tokenized_datasets["val"],
    collate_fn=data_collator,  # padding, teacher forcing
    shuffle=False,
    batch_size=BATCH_SIZE
)

# The number of training steps is the number of samples in the dataset, divided by the batch size then multiplied
# by the total number of epochs. Note that the tf_train_dataset here is a batched tf.data.Dataset,
# not the original Hugging Face Dataset, so its len() is already num_samples // batch_size.
num_train_steps = len(tf_train_dataset) * EPOCHS

optimizer, schedule = create_optimizer(
    init_lr=5.6e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)

model.compile(optimizer=optimizer)

# Train in mixed-precision float16
tf.keras.mixed_precision.set_global_policy("mixed_float16")

model.fit(
    tf_train_dataset, validation_data=tf_eval_dataset, epochs=EPOCHS
)

model.save_pretrained(f"transformer-model-epochs-{EPOCHS}")
tokenizer.save_pretrained(f"transformer-model-epochs-{EPOCHS}")
