import pickle
import numpy as np

DATASET_RANGE = 5
EMBEDDING_DIM = 100

with open('embedding_matrix.pickle', 'rb') as handle:
    full_embedding_matrix = pickle.load(handle)

with open(f'./../pickles-for-testing/{DATASET_RANGE}/word2idx_inputs.pkl', 'rb') as pickle_handle:
    word2idx_inputs = pickle.load(pickle_handle)

# print(word2idx_inputs)
num_tokens = len(word2idx_inputs) + 1
embedding_dim = 100
hits = 0
misses = 0

embedding_matrix = np.zeros((num_tokens, embedding_dim))

embedding_vector = full_embedding_matrix.get("microsoft")
print(embedding_vector)

for word, i in word2idx_inputs.items():
    embedding_vector = full_embedding_matrix.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
        print(word)

print("Converted %d words (%d misses)" % (hits, misses))

with open(f'embedding_matrix_100d_range_{DATASET_RANGE}.pickle', 'wb') as handle:
    pickle.dump(embedding_matrix, handle)
