import pickle
import json
import random

import keras
import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.layers import Dense, Embedding, Input, LSTM
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from rouge_score import rouge_scorer

import load_dataset as ld
from DataGenerator import DataGenerator

tf.compat.v1.enable_eager_execution()

# configuration
BATCH_SIZE = 500  # batch size for training
EPOCHS = 500  # number of epochs to train for
LATENT_DIM = 256  # latent dimensionality of the encoding space
# MAX_SEQUENCE_LENGTH = 1000 # with extract dataset
MAX_SEQUENCE_LENGTH = 250 # with headlines dataset
MAX_NUM_WORDS = 40000
EMBEDDING_DIM = 100
SAVE_FREQUENCY = 100  # for saving models during training

# load the data - original text
input_texts = []
# summarized text - original abstract
target_texts = []
# with the start of sentence token
target_texts_inputs = []

# load in the data
TEST_DATASET = "test-dataset/"
TRAIN_DATASET = "train-dataset/"
# 5, 10, 15, 20, 25
DATASET_RANGE = 5


# dataset with subset of sentences and abstracts
# data = ld.read_dataset(TRAIN_DATASET, DATASET_RANGE)
#
# # no need to do it every time
# with open('train-dataset.pkl', 'wb') as pkl_handle:
#     pickle.dump(data, pkl_handle)

# dataset with abstracts and titles
with open("./helper/headlines/headlines_dataset.json", encoding="utf-8") as f:
    all_data = json.loads(f.read())

train_dataset = all_data["data"]["train"]["data"]
val_dataset = all_data["data"]["val"]["data"]
data = train_dataset + val_dataset
random.shuffle(data)

# Iterating through the json list
for i in data:
    input_text, summarization = i['original'], i['summary']
    # make the target input and output
    input_text = input_text.replace("-", "")  # !!!!!for abstracts!
    # print(input_text)
    # print(summarization)
    # print(">>")
    target_text = summarization + ' <eos>'
    # used for teacher forcing
    target_texts_input = '<sos> ' + summarization

    input_texts.append(input_text)
    target_texts.append(target_text)
    target_texts_inputs.append(target_texts_input)

print('Num samples:', len(input_texts))

# tokenize the inputs
tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)

# pickle tokenizer_inputs, store them for testing
# no need to do it every time
# with open(f'pickles-for-testing/{DATASET_RANGE}/tokenizer.pkl', 'wb') as pkl_handle:
#     pickle.dump(tokenizer_inputs, pkl_handle)

# with open(f'pickles-for-testing-headlines/tokenizer.pkl', 'wb') as pkl_handle:
#     pickle.dump(tokenizer_inputs, pkl_handle)

input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
# get the word to index mapping for input
word2idx_inputs = tokenizer_inputs.word_index
print('Found %s unique input tokens' % len(word2idx_inputs))

# no need to do it every time
# with open(f'pickles-for-testing/{DATASET_RANGE}/word2idx_inputs.pkl', 'wb') as pkl_handle:
#     pickle.dump(word2idx_inputs, pkl_handle)

# with open(f'pickles-for-testing-headlines/word2idx_inputs.pkl', 'wb') as pkl_handle:
#     pickle.dump(word2idx_inputs, pkl_handle)

# determine maximum length in input sequence
max_len_input = max(len(s) for s in input_sequences)
print('Max len input sequence:', max_len_input)

# tokenize the outputs
# don't filter out special characters - otherwise <sos> won't appear
tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)  # inefficient, oh well
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)

# get the word to index mapping for output
word2idx_outputs = tokenizer_outputs.word_index
print('Found %s unique output tokens' % len(word2idx_outputs))

# no need to do it everytime
# with open(f'pickles-for-testing/{DATASET_RANGE}/word2idx_outputs.pkl', 'wb') as pkl_handle:
#     pickle.dump(word2idx_outputs, pkl_handle)

# with open(f'pickles-for-testing-headlines/word2idx_outputs.pkl', 'wb') as pkl_handle:
#     pickle.dump(word2idx_outputs, pkl_handle)

# number of outputs for later
# + 1 because indexing starts at 1 in keras
num_words_output = len(word2idx_outputs) + 1

# determine maximum length of output sequence
max_len_target = max(len(s) for s in target_sequences)
print('Max len target sequence:', max_len_target)

# pad the sequences
# add zeros at the beginning
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)
print('encoder_inputs.shape:', encoder_inputs.shape)
# print('encoder_inputs[0]:', encoder_inputs[0])

# add zeros at the end
decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')
print('decoder_inputs.shape:', decoder_inputs.shape)
# print('decoder_inputs[0]:', decoder_inputs[0])

# add zeros at the end
decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')

num_words = len(word2idx_inputs) + 1
with open(f'embedding/embedding_matrix_100d_range_{DATASET_RANGE}.pickle', 'rb') as pickle_handle:
    embedding_matrix = pickle.load(pickle_handle)

embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    input_length=MAX_SEQUENCE_LENGTH,
    # embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
    trainable=True
)
print('embedding done')

# create targets, since we cannot use sparse
# categorical cross entropy when we have sequences
# decoder_targets_one_hot = np.zeros(
#     (
#         len(input_texts),
#         max_len_target,
#         num_words_output
#     ),
#     dtype='float32'
# )

# assign the values
# for i, d in enumerate(decoder_targets):
#     for t, word in enumerate(d):
#         if word > 0:
#             decoder_targets_one_hot[i, t, word] = 1

# build the model
encoder_inputs_placeholder = Input(shape=(max_len_input,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(LATENT_DIM, return_state=True, dropout=0.5)
encoder_outputs, h, c = encoder(x)

# keep the states to pass into decoder
encoder_states = [h, c]

# set up the decoder, using [h, c] as initial state
decoder_inputs_placeholder = Input(shape=(max_len_target,))
decoder_embedding = Embedding(num_words_output, EMBEDDING_DIM, trainable=True)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

# since the decoder is a "to-many" model we want to have:
# return_sequences = True
decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(
    decoder_inputs_x,
    initial_state=encoder_states
)

# final dense layer for predictions
decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
print('create model')

model = Model(
    [encoder_inputs_placeholder, decoder_inputs_placeholder],
    decoder_outputs
)
print('compiling model...')

for i, v in enumerate(model.layers):
    print(i, v)

model.summary()

# # model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
# model.compile(optimizer='rmsprop', loss=custom_loss, metrics=[acc])

print('compiling model done')
print("ENCODER INPUTS SHAPE:", encoder_inputs.shape)
print("DECODER INPUTS SHAPE:", decoder_inputs.shape)
print("DECODER TARGETS SHAPE:", decoder_targets.shape)

# split data to train and validation
# reorganize inputs and targets: {id-1: (encoder_inputs[0], decoder_inputs[0], decoder_targets[0]), ...}
# so that batches can be shuffled
idx = 0
dic_generator_train = {}
dic_generator_validation = {}
# split dic_generator into two parts
# encoder_inputs, decoder_inputs, decoder_targets - same length
split_index = int(len(encoder_inputs) * 0.8)
while idx < len(encoder_inputs):
    value = (encoder_inputs[idx], decoder_inputs[idx], decoder_targets[idx])
    # value = (encoder_inputs[idx], decoder_inputs[idx], decoder_targets_one_hot[idx])
    if idx < split_index:
        id_tuple = f'id-{idx}'
        dic_generator_train[id_tuple] = value
    else:
        id_tuple = f'id-val-{idx}'
        dic_generator_validation[id_tuple] = value
    idx += 1

# Generators
training_generator = DataGenerator(dic_generator_train, batch_size=BATCH_SIZE, shuffle=True)
# print(training_generator)
validation_generator = DataGenerator(dic_generator_validation, batch_size=BATCH_SIZE, shuffle=True)
# print(validation_generator)

# # map indexes back into real words
# # so we can view the results
idx2word_i = {v: k for k, v in word2idx_inputs.items()}
idx2word_o = {v: k for k, v in word2idx_outputs.items()}


model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
# model.compile(optimizer='rmsprop', loss=custom_loss, metrics=[acc])

checkpoint = keras.callbacks.ModelCheckpoint('s2s-{epoch:08d}.h5',
                                             monitor='val_loss',
                                             verbose=1,
                                             save_best_only=False,
                                             mode='auto',
                                             period=50)

es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# r = model.fit(
#     training_generator,
#     epochs=EPOCHS,
#     validation_data=validation_generator,
#     verbose=1,
#     callbacks=[checkpoint]
# )

r = model.fit(
    [encoder_inputs, decoder_inputs], decoder_targets,
    epochs=EPOCHS,
    validation_split=0.2,
    # callbacks=[es_callback]
)

model.save(f"headings_seq2seq_1L_e{EPOCHS}_model.h5")
model.save(f"headings_seq2seq_1L_e{EPOCHS}_model.keras")

print('fit model done')

# inference
# make predictions
# another model that can take in the rnn state and previous
# word as input and accept a T = 1 sequence

# the encoder will be stand-alone
# from this we will get our initial decoder state
encoder_model = Model(encoder_inputs_placeholder, encoder_states)
decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

# keep the states too, to be output of the sampling model
decoder_outputs, h, c = decoder_lstm(
    decoder_inputs_single_x,
    initial_state=decoder_states_inputs
)

decoder_states = [h, c]
decoder_outputs = decoder_dense(decoder_outputs)
# the sampling model
# inputs: y(t-1), h(t-1), c(t-1)
# outputs: y(t), h(t), c(t)
decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

encoder_model.save(f'headings_seq2seq_1L_e{EPOCHS}_encoder.keras')
decoder_model.save(f'headings_seq2seq_1L_e{EPOCHS}_decoder.keras')

import matplotlib.pyplot as plt

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig(f'LOSS_headings_seq2seq_1layer_custom_emb_b{BATCH_SIZE}_r{DATASET_RANGE}_e{EPOCHS}.png')  # Save the plot with a specific name
plt.close()  # Close the current plot to avoid overlapping if you reuse 'plt'

# accuracies
# plt.plot(r.history['acc'], label='acc')
# plt.plot(r.history['val_acc'], label='val_acc')
# plt.legend()
# plt.savefig(f'ACC_seq2seq_1layer_custom_emb_b{BATCH_SIZE}_r{DATASET_RANGE}_e{EPOCHS}.png')  # Save the plot with a specific name
# plt.close()  # Close the current plot to avoid overlapping if you reuse 'plt'
