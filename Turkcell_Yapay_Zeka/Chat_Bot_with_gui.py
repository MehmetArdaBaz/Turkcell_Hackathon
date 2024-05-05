import tkinter as tk
from tkinter import scrolledtext

import time

import random 
import json 
import pickle

from tkinter import*
from tkinter import simpledialog

import os

import numpy as np 

import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay

import subprocess


from tensorflow.keras.models import load_model




lemmatizer= WordNetLemmatizer()
ifadeler= json.loads(open("ifadeler.json").read())


words= pickle.load(open("words.pkl", "rb"))
classes= pickle.load(open("classes.pkl", "rb"))

model= load_model("Chat-Bot_model.h5")

def clean_up_sentence(sentence):
    sentence_words= nltk.word_tokenize(sentence)
    sentence_words= [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_off_words(sentence):
    sentence_words= clean_up_sentence(sentence)
    bag= [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    
    return np.array(bag)


def predict_class(sentence):
    bow = bag_off_words(sentence)
    res = model.predict(np.array([bow]))[0]
    
    ERROR_THRESHOLD = 0.50
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    print("Results:", results)  
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"ifade": classes[r[0]], "probability": r[1]})  

    print("Return List:", return_list)  
    
    return return_list


def get_response(intents_list, intents_json):
    try:
        tag = intents_list[0]["ifade"]
        list_of_intents = intents_json["ifadeler"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                return result

    except:
        category= category_prompt()
        update_json(message, category)    

def category_prompt():
    global category
    category = simpledialog.askstring("Input", "Ne demek istediğinizi anlayamadım. Lütfen kategorisini belirtin:")
    return category


def update_json(word, category): 
    os.remove("words.pkl")
    os.remove("classes.pkl")
    os.remove("Chat-Bot_model.h5")
    
    app.destroy()
    
    with open("ifadeler.json", "r+") as file:
        data = json.load(file)
        for intent in data["ifadeler"]:
            if intent["tag"] == category:
                intent["patterns"].append(word)
                break
        file.seek(0)  
        json.dump(data, file, indent=4) 
    
    
    
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
    
    subprocess.call(["python", "Chat_Bot_with_gui.py"])







app = tk.Tk()
app.title("Chat Bot")
app.geometry("400x500")  
app.configure(bg="#2C3E50")  


text_box = scrolledtext.ScrolledText(app, width=50, height=20, wrap=tk.WORD, bg="#34495E", fg="#FFFFFF", font=("Helvetica", 12))
text_box.pack(padx=10, pady=10)


input_box = tk.Entry(app, width=50, bg="#ECF0F1", fg="#2C3E50", font=("Helvetica", 12))
input_box.pack(padx=10, pady=10)

category = None

def send_message():
    global message
    message = input_box.get()
    message = message.lower()
    text_box.insert(tk.END, "Siz: " + message + "\n")
    input_box.delete(0, tk.END)

    error_threshold= 0.70
    
    intents = predict_class(message)
    if intents and float(intents[0]['probability']) < error_threshold:
        category = category_prompt()
        update_json(message, category)  
        
        
    res = get_response(intents, ifadeler)
    text_box.insert(tk.END, "Chat Bot: " + res + "\n")


    text_box.config(highlightbackground="#FFD700", highlightthickness=2)

send_button = tk.Button(app, text="Gönder", command=send_message, bg="#FFD700", fg="#FFFFFF", font=("Helvetica 12 bold"))
send_button.pack(padx=10, pady=10)


if __name__ == "__main__":
    app.mainloop()
