import pandas as pd
import numpy as np
import time
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Activation
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from keras.models import load_model
import matplotlib.pyplot as plt
with open('SDCNL/file1.csv', encoding="utf8", errors="ignore") as f_s:
    TweetID_file_s=pd.read_csv(f_s, header=None, names=['selftext','is_suicide'])

df=pd.DataFrame(TweetID_file_s,columns= ['selftext','is_suicide'])
##df = pd.read_excel(path, encoding='UTF-8')
##df = df[['selftext','is_suicide']]

df=df.iloc[1:,]
X = df['selftext'].fillna('').tolist()
X = [str(i) for i in X]
y = df['is_suicide'].fillna('').tolist()

RANDOM_STATE = 42

maxlen=100

# Tokenize and transform to integer index
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

seq = tokenizer.texts_to_sequences(X)
tweets= pad_sequences(seq, padding='post', maxlen=maxlen)



X=np.array(tweets)

y=tf.keras.utils.to_categorical(y, 2, dtype="float32")
y=np.array(y)

# Split train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=RANDOM_STATE)


##text_train=np.array(text_train)
##X_test=np.array(X_test)
##y_train=np.array(y_train)
##y_test=np.array(y_test)

# Tokenize and transform to integer index
#tokenizer = Tokenizer()
#tokenizer.fit_on_texts(text_train)

#X_train = tokenizer.texts_to_sequences(text_train)
#X_test = tokenizer.texts_to_sequences(text_test)



vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
maxlen = max(len(x) for x in X_train) # longest text in train set

# Add pading to ensure all vectors have same dimensionality
#X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
#X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
# Define CNN architecture

embedding_dim = 64

model = Sequential()
model.add(layers.Embedding(vocab_size, 64, input_length=maxlen))
model.add(layers.BatchNormalization())
model.add(layers.Activation('tanh'))
model.add(layers.SpatialDropout1D(0.1))
model.add(layers.Conv1D(32, kernel_size=3, activation="relu"))
model.add(layers.Bidirectional(LSTM(16, return_sequences=False)))
#model.add(layers.attention(return_sequences=True))
model.add(layers.BatchNormalization())
model.add(layers.Activation('tanh'))
model.add(layers.Dropout(0.2))

##
##model.add(layers.GRU(128, dropout=0.2, return_sequences=True))
##model.add(layers.GRU(128, dropout=0.2, return_sequences=True))
###model.add(layers.MaxPool1D(5))
###model.add(layers.Conv1D(filters=128, kernel_size=3, padding="same", activation='relu'))
###model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(2, activation='sigmoid'))
##model.add(layers.SimpleRNN(64))
#model.add(layers.Dense(2, activation='sigmoid', use_bias=True, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), activity_regularizer=regularizers.l2(1e-5)))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print(model.summary())


model.save('CNN_model.h5')
# Fit model
history = model.fit(X_train, y_train,
                    batch_size=32, epochs = 3, validation_data=(X_test, y_test), shuffle=True)

loss, accuracy = model.evaluate(X_train, y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

plt.style.use('ggplot')
plt.show()

##def plot_history(history):
##    acc = history.history['acc']
##    val_acc = history.history['val_acc']
##    loss = history.history['loss']
##    val_loss = history.history['val_loss']
##    x = range(1, len(acc) + 1)
##
##    plt.figure(figsize=(12, 5))
##    plt.subplot(1, 2, 1)
##    plt.plot(x, acc, 'b', label='Training acc')
##    plt.plot(x, val_acc, 'r', label='Validation acc')
##    plt.title('Training and validation accuracy')
##    plt.legend()
##    plt.subplot(1, 2, 2)
##    plt.plot(x, loss, 'b', label='Training loss')
##    plt.plot(x, val_loss, 'r', label='Validation loss')
##    plt.title('Training and validation loss')
##    plt.legend()
##    plt.show()
##
##
##plot_history(history)


##X_sample = ['this is a sample text']
##X_sample = tokenizer.texts_to_sequences(X_sample)
##X_sample = pad_sequences(X_sample, padding='post', maxlen=maxlen)
##
##y_sample = model.predict_classes(X_sample).flatten().tolist()
##
##print('Prediction: ',sample)
##
##
##
#model.save('CNN_model.h5')
#model_load = load_model('my_model.h5')