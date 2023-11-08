from rouge import Rouge
import pickle
import numpy as np
from keras.models import load_model
from keras.utils import pad_sequences
import load_dataset as ld
from rouge_score import rouge_scorer
import json

RANGE = 5

# load the data
input_texts = []
target_texts = []
max_len_input = 236
max_len_target = 160

# this is for papers
# with open(f'./../pickles-for-testing/{RANGE}/tokenizer.pkl', 'rb') as pickle_handle:
#     tokenizer_inputs = pickle.load(pickle_handle)
#
# # load dictionaries
# with open(f'./../pickles-for-testing/{RANGE}/word2idx_inputs.pkl', 'rb') as pickle_handle:
#     word2idx_inputs = pickle.load(pickle_handle)
#
# with open(f'./../pickles-for-testing/{RANGE}/word2idx_outputs.pkl', 'rb') as pickle_handle:
#     word2idx_outputs = pickle.load(pickle_handle)
# print(word2idx_outputs)


# this is for headers
with open(f'./../pickles-for-testing-headlines/tokenizer.pkl', 'rb') as pickle_handle:
    tokenizer_inputs = pickle.load(pickle_handle)

# load dictionaries
with open(f'./../pickles-for-testing-headlines/word2idx_inputs.pkl', 'rb') as pickle_handle:
    word2idx_inputs = pickle.load(pickle_handle)

with open(f'./../pickles-for-testing-headlines/word2idx_outputs.pkl', 'rb') as pickle_handle:
    word2idx_outputs = pickle.load(pickle_handle)

print("loading tokenizers done")

# load in the data for papers
# data = ld.read_dataset('./../test-dataset/', RANGE)

# load in the data for headlines
with open("./../helper/headlines/headlines_dataset.json", encoding="utf-8") as f:
    all_data = json.loads(f.read())
train_dataset = all_data["data"]["train"]["data"]
val_dataset = all_data["data"]["val"]["data"]
# data = train_dataset + val_dataset

data = all_data["data"]["test"]["data"]

# Iterating through the json list
for i in data:
    input_text, summarization = i['original'], i['summary']
    # make the target input and output
    input_text = input_text.replace("-", "")  # !!!!!for abstracts!
    target_text = summarization + ' <eos>'
    # used for teacher forcing
    target_texts_input = '<sos> ' + summarization

    input_texts.append(input_text)
    target_texts.append(target_text)

print("loading data done")
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)

# make predictions
# we need to create another model that can take in
# the rnn state and previous word as input
# and accept a T=1 sequence

model_folder_path = "./../trained-models/headings_seq2_seq_3L_e200"
encoder_model_path = f"{model_folder_path}/headings_seq2seq_3L_e200_encoder.keras"
decoder_model_path = f"{model_folder_path}/headings_seq2seq_3L_e200_decoder.keras"


rouge = Rouge()
encoder_model = load_model(encoder_model_path, compile=False)
decoder_model = load_model(decoder_model_path, compile=False)
print("loading models done")

idx2word_eng = {v: k for k, v in word2idx_inputs.items()}
idx2word_trans = {v: k for k, v in word2idx_outputs.items()}

for k, test_text in enumerate(input_texts):
    if k == 35:
        break
    print(f"> Processing {k}...")
    output_file_path = f"{model_folder_path}/test-results.txt"
    output_file = open(output_file_path, "a", encoding="utf-8")
    # map indexes back into real world
    # so we can view the results
    def decode_sequence(input_seq):
        # print(input_seq)
        # encode the input as state vectors
        states_value = encoder_model.predict(input_seq, verbose=None)
        # generate empty target sequence of length 1
        target_seq = np.zeros((1, 1))
        # populate the first character of target sequence with the start character
        # NOTE: tokenizer lower-cases all words
        target_seq[0, 0] = word2idx_outputs['<sos>']

        # if we get this we break
        eos = word2idx_outputs['<eos>']

        # create the translation
        output_sentence = []
        while len(output_sentence) != max_len_target:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value,
                verbose=None
            )

            # get next word
            idx = np.argmax(output_tokens[0, 0, :])

            # end sentence of eos
            if eos == idx:
                break

            if idx > 0:
                word = idx2word_trans[idx]
                # print(word)
                output_sentence.append(word)

            # update the decoder input
            # which is the word we just generated
            target_seq[0, 0] = idx

            # update states
            states_value = [h, c]
            # states_value = [h] # gru
        return ' '.join(output_sentence)

    input_seq = encoder_inputs[k:k + 1]
    summarization_predicted = decode_sequence(input_seq)
    summarization_predicted = summarization_predicted.replace("-", "")
    summarization_true = target_texts[k].replace("<eos>", "")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(summarization_true, summarization_predicted)
    print(f"INPUT\n{input_texts[k]}")
    output_s = f"PREDICTION\n{summarization_predicted}"
    input_s = f"TARGET\n{summarization_true}"
    print(input_s)
    output_file.write(f"{input_s}\n")
    print()
    print(output_s)
    output_file.write(f"{output_s}\n")
    print()
    for key in scores:
        pair = f"{key}: {scores[key]}"
        print(pair)
        output_file.write(f"{pair}\n")
    print('--------------')
    output_file.write("--------------\n")
    output_file.close()
