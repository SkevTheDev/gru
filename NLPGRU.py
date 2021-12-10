import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from Utils import Utils
from NLPModel import NLPModel

#https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/20_Natural_Language_Processing.ipynb

trainModel = True  # set to True if you want to train

def main():
    print(tf.keras.__version__)
    print(tf.__version__)
    model = None  # three layer GRU model with binary classifier
    utils = Utils()
    
    #--------------read training data------------------
    x_train, x_test, test_y, train_y = utils.read_data()
    data_text = x_train + x_test
    print('training + test data shape = ' + str(data_text.shape))

    #------------tokenize data------------------------
    num_words = 10000 #- set it to None to use all words
    num_tokens, tokenizer, x_train_tokens, x_test_tokens = utils.tokenize_data(num_words, data_text, x_train, x_test)
    print(len(num_tokens))
    # num_tokens is an array of size 50000 (25000 for test, 25000 for train) that contains the
    # number of tokens for each review
    
    #-----------pad sequences---------------------
    # pad_sequences will pad a sequence if its length is less than (mean seq len + 2*std dev)=580
    # otherwise it will truncate the sequence.
    max_tokens, x_train_pad, x_test_pad = utils.pad_sequences(num_tokens, x_train_tokens, x_test_tokens)
    print("max tokens in a review = " + str(max_tokens)) # 580
    print(x_train_pad.shape) # (25000,580)
    print(x_test_pad.shape)  # (25000,580)
    #print(x_train_tokens[1])
    #print(x_train_pad[1])
    
    #--------------------------------------------------
    # tokenizer inverse map
    idx = tokenizer.word_index
    inverse_map = dict(zip(idx.values(), idx.keys()))
    print(len(inverse_map))  # 88582  - number of words encountered
    print(utils.tokens_to_string(x_train_tokens[82],inverse_map)) # print review from list of tokens

    #-------------------create or load model-------------------
    nlpModel = NLPModel() 
    model = None
    if trainModel == False:
        model = tf.keras.models.load_model('mymodel.h5')
    if trainModel == True:
        if num_words is None:
            num_words = len(tokenizer.word_index)
        model =  nlpModel.create_and_trainmodel(num_words, max_tokens, x_train_pad, train_y, 8, 3)
        # 8 is the embedding size, 3 is the number of epochs, 4 epochs overfits the data
    result = model.evaluate(x_test_pad, test_y)
    print("Accuracy (on 25000 test cases): {0:.2%}".format(result[1]))  # around 87.5% takes a while
    print(model.summary())

    #---------------------test model----some misclassified examples------
    y_pred = model.predict(x=x_test_pad[0:1000])
    y_pred = y_pred.T[0]
    cls_pred = np.array([1.0 if p>0.5 else 0.0 for p in y_pred])
    cls_true = np.array(test_y[0:1000])
    incorrect = np.where(cls_pred != cls_true)
    incorrect = incorrect[0]
    print('incorrectly classified out of 1000 test cases ='+ str(len(incorrect)))
    
    idx = incorrect[0]  # index of first misclassied text
    text = x_test[idx]
    print('mis classified\n' + text)
    print('predicted label ' + str(y_pred[idx]))
    print('true label ' + str(cls_true[idx]))

    #---------------tet model on new data---------------------
    text1 = "This movie is fantastic! I really like it because it is so good!"
    text2 = "Good movie!"
    text3 = "Maybe I like this movie."
    text4 = "Meh ..."
    text5 = "If I were a drunk teenager then this movie might be good."
    text6 = "After I watched the movie, my conclusion is that it is a bad bad movie!"
    text7 = "absolutely not a good movie!"
    text8 = "This movie really sucks! Can I get my money back please?"
    text9 = "this movie has no plot, acting is poor."
    text10 = "incoherent and poor plot."
    texts = [text1, text2, text3, text4, text5, text6, text7, text8, text9, text10]

    tokens = tokenizer.texts_to_sequences(texts)
    pad = 'pre'
    tokens_pad = pad_sequences(tokens, maxlen=max_tokens,
                               padding=pad, truncating=pad)
    print(tokens_pad.shape)
    res = model.predict(tokens_pad)
    print('--prediction result on 8 test custom cases--')
    print(res)
    
    # -------------------examine embeddings----------------------
    layer_embedding = model.get_layer('layer_embedding')
    weights_embedding = layer_embedding.get_weights()[0]
    #print(weights_embedding.shape)
    token_good = tokenizer.word_index['good']
    #print(token_good)
    token_great = tokenizer.word_index['great']
    #print(token_great)
    print(weights_embedding[token_good])
    print(weights_embedding[token_great])

    token_bad = tokenizer.word_index['bad']
    token_horrible = tokenizer.word_index['horrible']
    print(weights_embedding[token_bad])

    #--------------------print distances from words-------------------
    utils.print_sorted_words('great', tokenizer,weights_embedding, inverse_map,metric='cosine')
    utils.print_sorted_words('worst', tokenizer, weights_embedding, inverse_map,metric='cosine')

if __name__ == "__main__":
    sys.exit(int(main() or 0))