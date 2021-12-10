import tensorflow as tf

class NLPModel(object):
    def create_and_trainmodel(self, num_words, max_tokens, x_train_pad, train_y, embed_size, numepochs):
        model = tf.keras.Sequential()
        embedding_size = embed_size
        model.add(tf.keras.layers.Embedding(input_dim=num_words,
                            output_dim=embedding_size,
                            input_length=max_tokens,
                            name='layer_embedding'))

        model.add(tf.keras.layers.GRU(units=16, return_sequences=True))
        model.add(tf.keras.layers.GRU(units=8, return_sequences=True))
        model.add(tf.keras.layers.GRU(units=4))

        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        model.summary()
        model.fit(x_train_pad, train_y,
              validation_split=0.05, epochs=numepochs, batch_size=64)
        model.save('mymodel.h5')
        print('training done... - model saved')
        return model