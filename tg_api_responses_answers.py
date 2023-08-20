
import sys
import requests
import json
import pandas as pd
import numpy as np
# from transformers import pipeline
import logging
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from nn_model import load_trained_model_and_components
from nn_model import get_data
import pickle


token = '6376335129:AAGAR_Tg2GfkTl5RU6Y82oF6_3x6yKnqa_4'
base_url = f'https://api.telegram.org/bot{token}/'


def get_updates(offset=None):
    url = base_url + 'getUpdates'
    params = {'offset': offset}
    response = requests.get(url, params=params)
    return response.json()


def send_message(chat_id, text):
    url = base_url + 'sendMessage'
    data = {'chat_id': chat_id, 'text': text}
    response = requests.post(url, data=data)
    return response.json()


def classify_message(message, model, tokenizer, label_encoder):
    message_vector = tokenizer([message])
    predicted_probs = model.predict(message_vector)
    predicted_class_index = np.argmax(predicted_probs)
    predicted_category = label_encoder.inverse_transform([predicted_class_index])[0]
    return predicted_category


def determine_response(predicted_category):
    responses = {
        'приветствие': 'Привет! Как я могу Вам помочь?',
        'прощание': 'До свидания! Если у Вас еще будут вопросы - обращайтесь.',
        'сколько тебе лет': 'Я - бот, мой возраст не имеет значения.',
        'кто ты': 'Я бот, созданный для общения и помощи.',
        'как дела': 'У меня все отлично, а как Ваши дела?',
    }
    return responses.get(predicted_category, 'Простите, я не могу понять Ваш вопрос.')


def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.debug("Code starting")
    offset = None
    model, tokenizer, label_encoder = load_trained_model_and_components()

    while True:
        updates = get_updates(offset)
        if updates.get('result'):
            for update in updates['result']:
                offset = update['update_id'] + 1
                chat_id = update['message']['chat']['id']
                message_text = update['message']['text']
                logging.debug(f'Message received: {message_text}')
                predicted_category = classify_message(message_text, model, tokenizer, label_encoder)
                response = determine_response(predicted_category)
                send_message(chat_id, response)


main()

