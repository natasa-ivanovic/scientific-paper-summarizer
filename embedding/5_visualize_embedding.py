import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

embedding_file = "custom_glove_srb_100d.txt"
embeddings = {}
with open(embedding_file, "r", encoding="utf-8") as file:
    lines = file.readlines()[1:]  # Skip the first line
    for line in lines:
        parts = line.split()
        word = parts[0]
        vector = np.array([float(val) for val in parts[1:]])
        embeddings[word] = vector

# Determine the batch size
batch_size = 100

# Get all the words from your embeddings
all_words = list(embeddings.keys())

# Loop through the words in batches of 'batch_size'
for i in range(0, len(all_words), batch_size):
    # Get a batch of words
    batch_words = all_words[i:i + batch_size]

    # Extract the corresponding vectors for the batch
    vectors = np.array([embeddings[word] for word in batch_words])

    # Apply t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, perplexity=10, n_iter=300)
    vectors_2d = tsne.fit_transform(vectors)

    # Create and save a plot for this batch
    plt.figure(figsize=(10, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])

    for j, word in enumerate(batch_words):
        x, y = vectors_2d[j]
        plt.annotate(word, (x, y), fontsize=12)

    plt.savefig(f"word_embeddings_batch_{i // batch_size}.png")  # Save the plot with a specific name
    plt.close()  # Close the current plot to avoid overlapping if you reuse 'plt'


