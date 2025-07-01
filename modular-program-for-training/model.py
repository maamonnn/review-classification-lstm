from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense

def train(X_train, X_test, y_train_class, y_test_class, y_train_sent, y_test_sent, num_class_targets, num_sent_targets):
    # Model klasifikasi
    model_classification = Sequential([
        Embedding(input_dim=500, output_dim=64),
        LSTM(64, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(num_class_targets, activation='softmax'),
    ])
    model_classification.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    history_classification = model_classification.fit(
        X_train, y_train_class,
        epochs=50,
        validation_data=(X_test, y_test_class),
        batch_size=16
    )

    # Model sentimen
    model_sentimen = Sequential([
        Embedding(input_dim=500, output_dim=64),
        LSTM(64, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(num_sent_targets, activation='softmax'),
    ])
    model_sentimen.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    history_sentimen = model_sentimen.fit(
        X_train, y_train_sent,
        epochs=50,
        validation_data=(X_test, y_test_sent),
        batch_size=16
    )

    return model_classification, model_sentimen, history_classification, history_sentimen
