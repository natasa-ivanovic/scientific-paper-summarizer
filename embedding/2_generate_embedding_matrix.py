import numpy as np
import pickle

embeddings_index = {}
with open('custom_glove_srb_100d.txt', encoding='utf-8') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        print(f'{word}: {coefs}')
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

with open('embedding_matrix.pickle', 'wb') as handle:
    pickle.dump(embeddings_index, handle)
