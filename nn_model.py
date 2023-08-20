import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pickle


def get_data(spreadsheet_id, sheet_name):
    json_key_file = f'/Users/mihailtarasov/Documents/bot-infrastructure-396318-22bf9bdb6cfe.json'
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(json_key_file, scope)
    client = gspread.authorize(credentials)
    spreadsheet = client.open_by_key(spreadsheet_id)
    worksheet = spreadsheet.worksheet(sheet_name)
    data = worksheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])
    return df


def load_trained_model_and_components():
    train = get_data('1mXZZj7uxOBDrucuqPM7_2MPEAJufqmIdVGKMpUPMf8Y', 'train_set')
    texts = train['text'].values
    labels = train['label'].values

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    tokenizer = tf.keras.layers.TextVectorization(output_sequence_length=5)
    tokenizer.adapt(texts)
    max_length = tokenizer.get_config()['max_tokens']
    text_vectors = tokenizer(texts)
    num_classes = len(set(encoded_labels))

    model = tf.keras.Sequential([
                                 tf.keras.layers.Input(shape=(text_vectors.shape[1],)),
                                 tf.keras.layers.Dense(128, activation='relu'),
                                 tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(text_vectors, encoded_labels, epochs=1000, batch_size=32)

    # model.save('trained_model')  # Save the entire model

    return model, tokenizer, label_encoder

