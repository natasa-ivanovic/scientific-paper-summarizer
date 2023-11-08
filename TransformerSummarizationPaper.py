import evaluate
import tensorflow as tf
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import TFAutoModelForSeq2SeqLM
from transformers import create_optimizer

# load in the data
TEST_DATASET = "test-dataset/"
TRAIN_DATASET = "train-dataset/"
EPOCHS = 20
BATCH_SIZE = 8
# 5, 10, 15, 20, 25
RANGE = 10

# model_checkpoint = "facebook/mbart-large-50"
model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# INPUT - extracted subset of paper

# range 5
# Max num of words: 685
# Min num of words: 73
# Average num of words: 184.46740220661985
# Median num of words: 171.0

# range 10
# Max num of words: 1009
# Min num of words: 154
# Average num of words: 343.19057171514544
# Median num of words: 327.0

# TARGET - abstracts

# Max num of words: 241
# Min num of words: 10
# Average num of words: 63.99247743229689
# Median num of words: 58.0
# Low Median num of words: 58
# High Median num of words: 58

max_input_length = 1009
max_target_length = 241

data_files = {
    "train": f"./helper/papers/{RANGE}/dataset_train_range_{RANGE}.json",
    "val": f"./helper/papers/{RANGE}/dataset_val_range_{RANGE}.json",
    "test": f"./helper/papers/{RANGE}/dataset_test_range_{RANGE}.json"
}

datasets = load_dataset("json", data_files=data_files, field="data")

print(datasets)


def preprocess_function(examples):
    model_inputs = tokenizer(
        # abstracts
        examples["original"],
        max_length=max_input_length,
        truncation=True,
        padding=True
    )
    labels = tokenizer(
        # titles
        examples["summary"],
        max_length=max_target_length,
        truncation=True,
        padding=True
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
# features = [tokenized_datasets["train"][i] for i in range(2)]
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

history = model.fit(
    tf_train_dataset,
    validation_data=tf_eval_dataset,
    epochs=EPOCHS
)

model.save_pretrained(f"paper-transformer-model-epochs-{EPOCHS}-r{RANGE}")
tokenizer.save_pretrained(f"paper-transformer-model-epochs-{EPOCHS}-r{RANGE}")

train_losses = history.history['loss']
val_losses = history.history['val_loss']

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')

# Save the plot as a PNG file
plt.savefig(f"paper-transformer-model-epochs-{EPOCHS}-r{RANGE}-loss_plot.png")

# Close the plot to free up resources
plt.close()
