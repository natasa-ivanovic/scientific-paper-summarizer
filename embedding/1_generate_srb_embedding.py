import os
import re
import nltk
from gensim.models import Word2Vec


directory = "./../combined-dataset"
tokenized_sentences = []

for folder in os.listdir(directory):
    f = os.path.join(directory, folder)
    filename = f"{f}\\{folder}-processed.txt"
    with open(filename, encoding="utf-8") as my_file:
        file_content = my_file.read()
        sentences = nltk.sent_tokenize(file_content)
        for sentence in sentences:
            processed_sentence = re.sub('[^a-zA-ZščćžđŠČĆŽĐ ]', '', sentence)
            processed_sentence = processed_sentence.lower()
            tokens = processed_sentence.split(" ")
            processed_tokens = [token for token in tokens if len(token) > 1]
            if len(processed_tokens) > 0:
                tokenized_sentences.append(processed_tokens)

for sentence in tokenized_sentences:
    print(sentence)

model = Word2Vec(sentences=tokenized_sentences,
                 sg=1,
                 vector_size=100,
                 workers=4)

model.wv.save_word2vec_format('custom_glove_srb_100d.txt')


# print(model.vector_size)
# print(model.most_similar('fotografiju'))
