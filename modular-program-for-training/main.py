import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

import preprocessing
import model

# Load data
df = pd.read_csv('/Users/ferdianadham/Downloads/review-classification-lstm/modular-program-for-training/data_train.csv')

# Preprocessing teks
df['Review'] = df['Review'].map(lambda x: x.lower() if isinstance(x, str) else x)
df['Review'] = df['Review'].apply(preprocessing.remove_symbol)
df['Review'] = df['Review'].apply(preprocessing.stem_text)

# Konversi teks ke urutan angka
X = preprocessing.text_to_sequences(df['Review'])

# Encode target klasifikasi dan sentimen
y_class = preprocessing.class_encoder(df['Label Review'])
y_sent = preprocessing.sentimen_encoder(df['Kepuasan Umum/Sentimen'])

# Split data
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
_, _, y_train_sent, y_test_sent = train_test_split(X, y_sent, test_size=0.2, random_state=42)

# Train model
num_class_targets = len(set(y_class))
num_sent_targets = len(set(y_sent))

model_classification, model_sentimen, hist_class, hist_sent = model.train(
    X_train, X_test, y_train_class, y_test_class, y_train_sent, y_test_sent,
    num_class_targets, num_sent_targets
)

# Save model
joblib.dump(preprocessing.class_encoderencoder, 'class_encoder.pkl')
joblib.dump(preprocessing.sentimen_encoder, 'sentimen_encoder.pkl')
joblib.dump(preprocessing.tokenizer, 'tokenizer.json')

model.save('model_classification.keras')
model_sentimen.save('model_classification.keras')