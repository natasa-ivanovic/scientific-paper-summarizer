import pickle
import keras
import json
import random
from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM
from keras.preprocessing.text import Tokenizer
from keras.layers import TimeDistributed
from keras.utils import pad_sequences
import load_dataset as ld
import tensorflow as tf
from DataGenerator import DataGenerator

# configuration
BATCH_SIZE = 50  # batch size for training
EPOCHS = 2  # number of epochs to train for
LATENT_DIM = 256  # latent dimensionality of the encoding space
# MAX_SEQUENCE_LENGTH = 40000
MAX_SEQUENCE_LENGTH = 250
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
SAVE_FREQUENCY = 100

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

# data = ld.read_dataset(TRAIN_DATASET, DATASET_RANGE)

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
    input_text = input_text.replace("-", "")  # !!!!!for abstracts!
    # make the target input and output
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
# with open(f'pickles-for-testing/{DATASET_RANGE}/tokenizer.pkl', 'wb') as pkl_handle:
#     pickle.dump(tokenizer_inputs, pkl_handle)

with open(f'pickles-for-testing-headlines/tokenizer.pkl', 'wb') as pkl_handle:
    pickle.dump(tokenizer_inputs, pkl_handle)

input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
# get the word to index mapping for input
word2idx_inputs = tokenizer_inputs.word_index
print('Found %s unique input tokens' % len(word2idx_inputs))
# with open(f'pickles-for-testing/{DATASET_RANGE}/word2idx_inputs.pkl', 'wb') as pkl_handle:
#     pickle.dump(word2idx_inputs, pkl_handle)

with open(f'pickles-for-testing-headlines/word2idx_inputs.pkl', 'wb') as pkl_handle:
    pickle.dump(word2idx_inputs, pkl_handle)

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
# with open(f'pickles-for-testing/{DATASET_RANGE}/word2idx_outputs.pkl', 'wb') as pkl_handle:
#     pickle.dump(word2idx_outputs, pkl_handle)

with open(f'pickles-for-testing-headlines/word2idx_outputs.pkl', 'wb') as pkl_handle:
    pickle.dump(word2idx_outputs, pkl_handle)

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
# with open(f'embedding/embedding_matrix_100d_range_{DATASET_RANGE}.pickle', 'rb') as pickle_handle:
#     embedding_matrix = pickle.load(pickle_handle)
#
embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    input_length=MAX_SEQUENCE_LENGTH,
    # embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
    trainable=False
)
print('embedding done')

# build the model
# Encoder
encoder_inputs_placeholder = Input(shape=(max_len_input,))

# embedding layer
enc_emb = embedding_layer(encoder_inputs_placeholder)

# encoder lstm 1
encoder_lstm1 = LSTM(LATENT_DIM, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

# encoder lstm 2
encoder_lstm2 = LSTM(LATENT_DIM, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

# encoder lstm 3
encoder_lstm3 = LSTM(LATENT_DIM, return_state=True, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)
encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs_placeholder = Input(shape=(None,))

# embedding layer
dec_emb_layer = Embedding(num_words_output, EMBEDDING_DIM, trainable=True)
dec_emb = dec_emb_layer(decoder_inputs_placeholder)

decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.2)
decoder_outputs_s, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

# dense layer
decoder_dense = TimeDistributed(Dense(num_words_output, activation='softmax'))
decoder_outputs_ = decoder_dense(decoder_outputs_s)

print('create model')

# Define the model
model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder], decoder_outputs_)
model.summary()
# print('compiling model...')
#
# # for i, v in enumerate(model.layers):
# #     print(i, v)
#
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

print('compiling model done')
# print("ENCODER INPUTS SHAPE:", encoder_inputs.shape)
# print("DECODER INPUTS SHAPE:", decoder_inputs.shape)
# print("DECODER TARGETS SHAPE:", decoder_targets.shape)

# split data to train and validation
# reorganize inputs and targets: {id-1: (encoder_inputs[0], decoder_inputs[0], decoder_targets[0]), ...}
# so that batches can be shuffled
# idx = 0
# dic_generator_train = {}
# dic_generator_validation = {}
# # split dic_generator into two parts
# # encoder_inputs, decoder_inputs, decoder_targets - same length
# split_index = int(len(encoder_inputs) * 0.8)
# while idx < len(encoder_inputs):
#     value = (encoder_inputs[idx], decoder_inputs[idx], decoder_targets[idx])
#     if idx < split_index:
#         id_tuple = f'id-{idx}'
#         dic_generator_train[id_tuple] = value
#     else:
#         id_tuple = f'id-val-{idx}'
#         dic_generator_validation[id_tuple] = value
#     idx += 1
#
# # Generators
# training_generator = DataGenerator(dic_generator_train, batch_size=BATCH_SIZE, shuffle=True)
# print(training_generator)
# validation_generator = DataGenerator(dic_generator_validation, batch_size=BATCH_SIZE, shuffle=True)
# print(validation_generator)

checkpoint = keras.callbacks.ModelCheckpoint('s2s-{epoch:08d}.h5',
                                             monitor='val_loss',
                                             verbose=1,
                                             save_best_only=False,
                                             mode='auto',
                                             period=50)

# r = model.fit(
#     training_generator,
#     epochs=EPOCHS,
#     validation_data=validation_generator,
#     verbose=1,
#     callbacks=[checkpoint]
# )

r = model.fit(
    [encoder_inputs, decoder_inputs], decoder_targets,
    # batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    # callbacks=[es_callback]
)

print('fit model done')
model.save(f"headings_seq2seq_3L_e{EPOCHS}_model.h5")
model.save(f"headings_seq2seq_3L_e{EPOCHS}_model.keras")

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
decoder_inputs_single_x = dec_emb_layer(decoder_inputs_single)

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

encoder_model.save(f'headings_seq2seq_3L_e{EPOCHS}_encoder.keras')
decoder_model.save(f'headings_seq2seq_3L_e{EPOCHS}_decoder.keras')
