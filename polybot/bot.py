import json
import re

import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
import boto3
import requests


class Bot:

    def __init__(self, token, telegram_chat_url):
        # create a new instance of the TeleBot class.
        # all communication with Telegram servers are done using self.telegram_bot_client
        self.telegram_bot_client = telebot.TeleBot(token)

        # remove any existing webhooks configured in Telegram servers
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)

        # set the webhook URL
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', certificate=open('bot_cert.pem', 'r'),
                                             timeout=60)

        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        """
        Downloads the photos that sent to the Bot to `photos` directory (should be existed)
        :return:
        """
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path)
        )

    def handle_message(self, msg):
        """Bot Main message handler"""
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')


def format_prediction_result1(prediction_result):
    labels = prediction_result.get('labels', [])
    detected_objects = [label['class'] for label in labels]

    if detected_objects:
        detected_string = ', '.join(detected_objects)
        return f"Detected objects: {detected_string}"
    else:
        return "No objects detected."


def format_prediction_result(prediction_result_str):
    # Replace single quotes with double quotes to make it valid JSON
    valid_json_str = prediction_result_str.replace("'", '"')

    # Parse the JSON string into a dictionary
    try:
        prediction_result = json.loads(valid_json_str)
    except json.JSONDecodeError:
        logger.info(f"TYPE prediction_result_str: {type(prediction_result_str)}")
        return "Invalid prediction result format."

    labels = prediction_result.get('labels', [])
    logger.info(f"LABELS: {labels}")
    detected_objects = [label['class'] for label in labels]
    logger.info(f"DETECTED: {detected_objects}")

    if detected_objects:
        detected_string = ', '.join(detected_objects)
        return f"Detected objects: {detected_string}"
    else:
        return "No objects detected."


class ObjectDetectionBot(Bot):
    def __init__(self, token, telegram_chat_url, bucket_name, yolo5_service_url):
        super().__init__(token, telegram_chat_url)
        self.bucket_name = bucket_name
        self.yolo5_service_url = yolo5_service_url
        self.s3 = boto3.client('s3')

    def upload_photo_to_s3(self, photo_path):
        photo_name = os.path.basename(photo_path)
        self.s3.upload_file(photo_path, self.bucket_name, photo_name)
        return photo_name

    def get_yolo5_prediction(self, img_name):
        response = requests.post(f"{self.yolo5_service_url}/predict", params={'imgName': img_name})
        logger.info(f'Yolo5 service response status: {response.status_code}')
        logger.info(f'Yolo5 service response text: {response.text}')

        if response.status_code != 200:
            raise RuntimeError(f'Failed to get prediction from Yolo5 service: {response.text}')

        logger.info(f"TYPE of response.text: {type(response.text)}")

        valid_json_str = response.text.replace("'", '"')
        valid_json_str = re.sub(r'ObjectId\("([0-9a-fA-F]+)"\)', r'"\1"', valid_json_str)

        # Parse the JSON string into a dictionary
        try:
            prediction_result = json.loads(valid_json_str)
        except json.JSONDecodeError:
            return "Invalid prediction result format."

        labels = prediction_result.get('labels', [])
        logger.info(f"LABELS: {labels}")
        detected_objects = [label['class'] for label in labels]
        logger.info(f"DETECTED: {detected_objects}")

        if detected_objects:
            detected_string = ', '.join(detected_objects)
            return f"Detected objects: {detected_string}"
        else:
            return "No objects detected."

    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if self.is_current_msg_photo(msg):
            photo_path = self.download_user_photo(msg)
            self.send_text(msg['chat']['id'], 'I received the photo. Let me process it :)')

            try:
                img_name = self.upload_photo_to_s3(photo_path)
                logger.info(f'Uploading: photo uploaded to s3')

                prediction = self.get_yolo5_prediction(img_name)
                logger.info(f'Prediction: {prediction} type of it!!! {type(prediction)}')

                # Format the prediction result if 'labels' in prediction: labels = prediction['labels'] result_text =
                # "I detected the following objects:\n" + "\n".join( [ f"{label['class']} at ({label['cx']:.2f},
                # {label['cy']:.2f}) with size ({label['width']:.2f}, {label['height']:.2f})" for label in labels ] )
                # else: result_text = "Prediction result:\n" + "\n".join( [f"{key}: {value}" for key,
                # value in prediction.items()] )
                # x = format_prediction_result(prediction)
                # logger.info(f"TYPE of x: {type(x)}")
                self.send_text(msg['chat']['id'], prediction)

            except Exception as e:
                logger.error(f'Error handling message: {e}')
                self.send_text(msg['chat']['id'], f'Error :( : {e}')
