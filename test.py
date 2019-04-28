import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from data_structs import Vocabulary
import json,sys
import pandas as pd

def get_data():
    voc = Vocabulary(init_from_file="Voc")
    df=pd.read_csv('../../result/{}'.format(sys.argv[1]))
    smiles=df["Ligand SMILES"].tolist()
    label=df["label"].tolist()
    token_list = [voc.tokenize(mol) for mol in smiles]
    encode_list = [voc.encode(token) for token in token_list]
    print(set(voc.aa))
    # exit()
    with open("data.json","w") as f:
        json.dump({"encode_list":encode_list, "label":label},f)   
    return encode_list, label
try:
    with open("data.json","r") as f:
        r=json.load(f)
        encode_list=r["encode_list"]  
        label=r["label"]  
except:
    encode_list, label = get_data()

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 68 
X_train = encode_list
y_train = label

X_test = encode_list[5000:]
y_test = label[5000:]
#exit()
# truncate and pad input sequences
max_review_length = len(max(encode_list,key=len))
print(max_review_length)

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=30, batch_size=64)
# Final evaluation of the model
#scores = model.evaluate(X_test, y_test, verbose=0)
scores = model.evaluate(X_train, y_train, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
