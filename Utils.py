from tensorflow.keras.datasets import imdb
import numpy as np
from scipy.spatial.distance import cdist
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

class Utils(object):
    def read_data(self):
        #NUM_WORDS=1000 # was orig 1000,  only use top 1000 words
        INDEX_FROM=3   # word index offset  # was 3
        #train,test = imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)
        train,test = imdb.load_data()
        train_x,train_y = train
        test_x,test_y = test
        word_to_id = imdb.get_word_index()
        word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2
        word_to_id["<UNUSED>"] = 3

        id_to_word = {value:key for key,value in word_to_id.items()}
        print(' '.join(id_to_word[id] for id in train_x[0] ))
        x_train = np.empty(len(train_x), dtype = object)
        for i in range (0, len(train_x)):
              x_train[i] = ' '.join(id_to_word[id] for id in train_x[i])
        x_test = np.empty(len(test_x), dtype = object)
        print(x_test.shape)
        for i in range (0, len(test_x)):
          x_test[i] = ' '.join(id_to_word[id] for id in test_x[i])

        print("Train-set size: ", len(x_train))
        print("Test-set size:  ", len(x_test))
        return x_train, x_test, test_y, train_y
        # purpose of read-data is to read the imdb dataset
        # and return the x_train, train_y, x_test, test_y 
        # the x_train and x_test are 25000 reviews each     

    def tokenize_data(self, num_words, data_text, x_train, x_test):
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(data_text)

        if num_words is None:
            num_words = len(tokenizer.word_index)
        x_train_tokens = tokenizer.texts_to_sequences(x_train)
        print(x_train[1])
        print(np.array(x_train_tokens[1]))
        x_test_tokens = tokenizer.texts_to_sequences(x_test)
        num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
        num_tokens = np.array(num_tokens)
        print(len(num_tokens))
        return num_tokens, tokenizer, x_train_tokens, x_test_tokens
    # purpose: tokenize each unique word to a particular index
    
    def pad_sequences(self, num_tokens, x_train_tokens, x_test_tokens):
        #print(np.mean(num_tokens))
        #print(np.max(num_tokens))
        # The max number of tokens we will allow is set to the average plus 2 standard deviations.
        # This covers about 95% of the data-set.
        max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
        max_tokens = int(max_tokens)
        sum = np.sum(num_tokens < max_tokens) / len(num_tokens)
        print(sum)
        pad = 'pre'
        x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,
                                    padding=pad, truncating=pad)
        x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens,
                                   padding=pad, truncating=pad)
        return max_tokens, x_train_pad, x_test_pad

    def tokens_to_string(self,tokens, inverse_map):
        # Map from tokens back to words.
        words = [inverse_map[token] for token in tokens if token != 0]
        # Concatenate all words.
        text = " ".join(words)
        return text

    def print_sorted_words(self, word, tokenizer,weights_embedding, inverse_map,metric='cosine'):
        """
        Print the words in the vocabulary sorted according to their
        embedding-distance to the given word.
        Different metrics can be used, e.g. 'cosine' or 'euclidean'.
        """
        # Get the token (i.e. integer ID) for the given word.
        token = tokenizer.word_index[word]

        # Get the embedding for the given word. Note that the
        # embedding-weight-matrix is indexed by the word-tokens
        # which are integer IDs.
        embedding = weights_embedding[token]

        # Calculate the distance between the embeddings for
        # this word and all other words in the vocabulary.
        distances = cdist(weights_embedding, [embedding],
                          metric=metric).T[0]
    
        # Get an index sorted according to the embedding-distances.
        # These are the tokens (integer IDs) for words in the vocabulary.
        sorted_index = np.argsort(distances)
    
        # Sort the embedding-distances.
        sorted_distances = distances[sorted_index]
    
        # Sort all the words in the vocabulary according to their
        # embedding-distance. This is a bit excessive because we
        # will only print the top and bottom words.
        sorted_words = [inverse_map[token] for token in sorted_index
                        if token != 0]
        # Number of words to print from the top and bottom of the list.
        k = 10

        print("Distance from '{0}':".format(word))

        # Print the words with smallest embedding-distance.
        self._print_words(sorted_words[0:k], sorted_distances[0:k])
        print("...")

        # Print the words with highest embedding-distance.
        self._print_words(sorted_words[-k:], sorted_distances[-k:])

    # Helper-function for printing words and embedding-distances.
    def _print_words(self,words, distances):
        for word, distance in zip(words, distances):
            print("{0:.3f} - {1}".format(distance, word))