import random
import json
import pickle
import numpy as np 

import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay




lemmatizer= WordNetLemmatizer()

ifadeler= json.loads(open("ifadeler.json").read())

words= []
classes= []
documents= []
ignore_letters= ["?", "!", ".", ","]


for ifade in ifadeler["ifadeler"]:
    if "patterns" in ifade: 
        for pattern in ifade["patterns"]:
            word_list= nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, ifade["tag"]))
            if ifade["tag"] not in classes:
                classes.append(ifade["tag"])
            

words= [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words= sorted(set(words))


classes= sorted(set(classes))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))


training= []
output_empty= [0] * len(classes)

for document in documents:
    bag= []
    word_patterns= document[0]
    word_patterns= [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row= list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])
    

random.shuffle(training)

X_train = np.array([bag for bag, _ in training])
y_train = np.array([output_row for _, output_row in training])



model= Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation="softmax"))


initial_learning_rate = 0.01 
decay_rate = 0.001  
decay_steps = 1000

learning_rate_schedule = ExponentialDecay(
    initial_learning_rate, decay_steps, decay_rate, staircase=True
)

sgd = SGD(learning_rate=learning_rate_schedule, momentum=0.9, nesterov=True)


model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

hist= model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=5, verbose=1)
model.save("Chat-Bot_model.h5", hist)

print("Done!")  