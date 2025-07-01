import re
from sklearn.preprocessing import LabelEncoder
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class_encoder_global = LabelEncoder()
sentimen_encoder_global = LabelEncoder()
tokenizer_global = Tokenizer(num_words=500, oov_token="<OOV>")

def remove_symbol(doc):
    return re.sub(r'[^a-zA-Z0-9/]', ' ', doc)

def stem_text(doc):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(doc)

def class_encoder(labels):
    return class_encoder_global.fit_transform(labels)

def sentimen_encoder(sentimens):
    return sentimen_encoder_global.fit_transform(sentimens)

def text_to_sequences(docs):
    tokenizer_global.fit_on_texts(docs)
    sequences = tokenizer_global.texts_to_sequences(docs)
    return pad_sequences(sequences, maxlen=20)


def text_to_sequences(docs):
    tokenizer = Tokenizer(num_words=500, oov_token="<OOV>")
    tokenizer.fit_on_texts(docs)
    sequences = tokenizer.texts_to_sequences(docs)
    return pad_sequences(sequences, maxlen=20) 