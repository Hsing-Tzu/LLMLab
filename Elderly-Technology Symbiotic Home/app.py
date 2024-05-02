from flask import Flask, request, jsonify, render_template, Response
import requests

from flask import Flask, request, jsonify, render_template, Response
import os

import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

from PIL import Image
import io

global_image_cache = None


GOOGLE_API_KEY="AIzaSyDb_pJo02XNWlexXH6MHaahNrSfcl7Ftew"

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# 11 on
def turn_11_on():
  url = 'http://211.21.113.190:8155/api/webhook/-hFNoCcZKB31gtiZzhIabeI0d'
  headers = {
      'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }
  response = requests.post(url, headers=headers)
  return response.text

# 11 off
def turn_11_off():
  url = 'http://211.21.113.190:8155/api/webhook/-qtfn4apfmywr78fHHNZYiclU'
  headers = {
      'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }
  response = requests.post(url, headers=headers)
  return response.text

# 12 on
def turn_12_on():
    url = 'http://211.21.113.190:8155/api/webhook/-TJO7MQn5u--KlqSH4Mw2JHA7'
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }
    response = requests.post(url, headers=headers)
    return response.text

# 12 off
def turn_12_off():
    url = 'http://211.21.113.190:8155/api/webhook/-jAxX99jM2ghu4SVD29Ht8Flx'
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }
    response = requests.post(url, headers=headers)
    return response.text

# 13 on
def turn_13_on():
    url = 'http://211.21.113.190:8155/api/webhook/-GB_PabGDpQlRGcGChEhun6uj'
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }
    response = requests.post(url, headers=headers)
    return response.text

# 13 off
def turn_13_off():
    url = 'http://211.21.113.190:8155/api/webhook/1-3-off-fgzskQbLSxNVl3_SpTPlo7QP'
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }
    response = requests.post(url, headers=headers)
    return response.text

# 14 on
def turn_14_on():
    url = 'http://211.21.113.190:8155/api/webhook/-lkvcCfPU2wOmNLvhDjrEKpkb'
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }
    response = requests.post(url, headers=headers)
    return response.text

# 14 off
def turn_14_off():
    url = 'http://211.21.113.190:8155/api/webhook/-jfyKgpRXg6fgI9IXNIy0GNSn'
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }
    response = requests.post(url, headers=headers)
    return response.text

# 15 on
def turn_15_on():
    url = 'http://211.21.113.190:8155/api/webhook/-TqF-jZh-M8QqFBEnfgXEAxY7'
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }
    response = requests.post(url, headers=headers)
    return response.text

# 15 off
def turn_15_off():
    url = 'http://211.21.113.190:8155/api/webhook/--gONieqwhFofJ_6WjVAqxZtg'
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }
    response = requests.post(url, headers=headers)
    return response.text

# 16 on
def turn_16_on():
    url = 'http://211.21.113.190:8155/api/webhook/-d2Qi-uvQnpb_KcVcp_Jm9iJK'
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }

    response = requests.post(url, headers=headers)
    return response.text

# 16 off
def turn_16_off():
    url = 'http://211.21.113.190:8155/api/webhook/-pAW1x-AJO9s-b9JXDm89Dp_P'
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }

    response = requests.post(url, headers=headers)
    return response.text

def checkFall(file_storage):
    file_storage.seek(0)
    image = Image.open(file_storage)

    LAB_prompt = "I'd like you to help me confirm if there's anyone fallen in the following pictures. If someone has fallen, please return 1; otherwise, return 0."
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([LAB_prompt, image], stream=True)
    response.resolve()

    try:
        if response.parts and hasattr(response.parts[0], 'text'):
            fall_detected = int(response.parts[0].text)
            print(fall_detected)
            return fall_detected
        else:
            return 0 
    except Exception as e:
        print(f"Error processing the response: {e}")
        return 0
    
def BedLight(file_storage):
    file_storage.seek(0)
    image = Image.open(file_storage)
    LAB_prompt = "I'd like you to help me confirm if there's anyone lying on the bed in the following pictures. If someone is lying on the bed, please return 1; otherwise, return 0."
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([LAB_prompt, image], stream=True)
    response.resolve()

    try:
        if response.parts and hasattr(response.parts[0], 'text'):
            bed_presence = int(response.parts[0].text)
            print(bed_presence)
            return bed_presence
        else:
            return 0 
    except Exception as e:
        print(f"Error processing the response: {e}")
        return 0 

import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from zoneinfo import ZoneInfo

def storeFallData():
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']

    creds = Credentials.from_service_account_file('starry-hawk-419101-b66c43b71517.json', scopes=scope)
    client = gspread.authorize(creds)

    sheet_id = '1JgIbh7V6eAu2B22VRF6gRFmfQHZbu6bDIRw5737nLJ4'
    sheet = client.open_by_key(sheet_id).sheet1

    tz = ZoneInfo("Asia/Taipei")
    today_date = datetime.now(tz).strftime('%Y-%m-%d')
    today_time = datetime.now(tz).strftime('%H:%M:%S')

    intension = 1
    gas_state = 0
    alert_state = 1
    message_state = 1
    lamp_state = 2
    ac_state = 2

    row_data = [today_date, today_time, intension, gas_state, alert_state, message_state, lamp_state, ac_state]
    sheet.append_row(row_data)

    last_row_values = sheet.get_all_records()[-1]
    # print(last_row_values)[]
    return 0

def storeRAGData():
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']

    creds = Credentials.from_service_account_file('starry-hawk-419101-b66c43b71517.json', scopes=scope)
    client = gspread.authorize(creds)

    sheet_id = '1JgIbh7V6eAu2B22VRF6gRFmfQHZbu6bDIRw5737nLJ4'
    sheet = client.open_by_key(sheet_id).sheet1

    tz = ZoneInfo("Asia/Taipei")
    today_date = datetime.now(tz).strftime('%Y-%m-%d')
    today_time = datetime.now(tz).strftime('%H:%M:%S')

    intension = 6
    gas_state = 2
    alert_state = 2
    message_state = 2
    lamp_state = 2
    ac_state = 2

    row_data = [today_date, today_time, intension, gas_state, alert_state, message_state, lamp_state, ac_state]
    sheet.append_row(row_data)

    last_row_values = sheet.get_all_records()[-1]
    # print(last_row_values)
    return 0

def storeExerciseData():
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']

    creds = Credentials.from_service_account_file('starry-hawk-419101-b66c43b71517.json', scopes=scope)
    client = gspread.authorize(creds)

    sheet_id = '1JgIbh7V6eAu2B22VRF6gRFmfQHZbu6bDIRw5737nLJ4'
    sheet = client.open_by_key(sheet_id).sheet1

    tz = ZoneInfo("Asia/Taipei")
    today_date = datetime.now(tz).strftime('%Y-%m-%d')
    today_time = datetime.now(tz).strftime('%H:%M:%S')

    intension = 5
    gas_state = 2
    alert_state = 2
    message_state = 1
    lamp_state = 1
    ac_state = 1

    row_data = [today_date, today_time, intension, gas_state, alert_state, message_state, lamp_state, ac_state]
    sheet.append_row(row_data)

    last_row_values = sheet.get_all_records()[-1]
    # print(last_row_values)
    return 0

def storeFindData():
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']

    creds = Credentials.from_service_account_file('starry-hawk-419101-b66c43b71517.json', scopes=scope)
    client = gspread.authorize(creds)

    sheet_id = '1JgIbh7V6eAu2B22VRF6gRFmfQHZbu6bDIRw5737nLJ4'
    sheet = client.open_by_key(sheet_id).sheet1

    tz = ZoneInfo("Asia/Taipei")
    today_date = datetime.now(tz).strftime('%Y-%m-%d')
    today_time = datetime.now(tz).strftime('%H:%M:%S')

    intension = 9
    gas_state = 2
    alert_state = 2
    message_state = 2
    lamp_state = 1
    ac_state = 2

    row_data = [today_date, today_time, intension, gas_state, alert_state, message_state, lamp_state, ac_state]
    sheet.append_row(row_data)

    last_row_values = sheet.get_all_records()[-1]
    # print(last_row_values)
    return 0


def storeSleepData():
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']

    creds = Credentials.from_service_account_file('starry-hawk-419101-b66c43b71517.json', scopes=scope)
    client = gspread.authorize(creds)

    sheet_id = '1JgIbh7V6eAu2B22VRF6gRFmfQHZbu6bDIRw5737nLJ4'
    sheet = client.open_by_key(sheet_id).sheet1

    tz = ZoneInfo("Asia/Taipei")
    today_date = datetime.now(tz).strftime('%Y-%m-%d')
    today_time = datetime.now(tz).strftime('%H:%M:%S')

    intension = 10
    gas_state = 2
    alert_state = 2
    message_state = 1
    lamp_state = 0
    ac_state = 1

    row_data = [today_date, today_time, intension, gas_state, alert_state, message_state, lamp_state, ac_state]
    sheet.append_row(row_data)

    last_row_values = sheet.get_all_records()[-1]
    # print(last_row_values)
    return 0

import gspread
from google.oauth2.service_account import Credentials
import csv
import io

scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']

def export_recent_to_csv_string(sheet_id):
    credentials = Credentials.from_service_account_file('starry-hawk-419101-b66c43b71517.json', scopes=scope)

    gc = gspread.authorize(credentials)

    sheet = gc.open_by_key(sheet_id).sheet1

    fieldnames = sheet.row_values(1)

    fieldnames = [field for field in fieldnames if field in ['date', 'time', 'intension']]

    rows = sheet.get_all_values()
    recent_rows = rows[-100:]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in recent_rows[1:]:
        writer.writerow({field: row[fieldnames.index(field)] for field in fieldnames})

    csv_string = output.getvalue()
    output.close()
    return csv_string

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



app = Flask(__name__)

from io import BytesIO
@app.route('/capture', methods=['POST'])
def capture_image():
    global global_image_cache
    img = request.files.get('img')
    if img and allowed_file(img.filename):
        # turn_11_on()
        # turn_12_off()
        # turn_13_off()
        # turn_14_on()
        # turn_15_off()
        # turn_16_off()
        img_cache = BytesIO()
        img.save(img_cache)
        global_image_cache = img_cache

        returnValue = 0
        img_cache.seek(0)
        if checkFall(img_cache) == 1:
            returnValue = 1
            turn_11_off()
            turn_12_on()
            turn_13_on()
            storeFallData()
        # else:
        #     turn_12_off()

        img_cache.seek(0)
        if BedLight(img_cache) == 1:
            returnValue = 1
            turn_11_off()
            turn_13_on()
            turn_14_off()
            turn_15_on()

            storeSleepData()
        # else:
        #     turn_12_off()

        return jsonify({'status': returnValue})
    else:
        return jsonify({'error': 'Invalid or no image provided'}), 400

    
instruction = ""
    
def checkInputIntension(instruction):
    LAB_prompt = "Please help me confirm which of the following two intentions the content below represents. You only need to answer with a number (e.g., 1): Intention 1 (looking for an item): return the number 1; Intention 2 (wanting to exercise): return the number 2; Intention 3 (wanting to access historical data): return the number 3; Intention 4 (wanting to confirm an exercise movement): return the number 4. Content:"
    model = genai.GenerativeModel('gemini-pro')
    final_prompt = LAB_prompt + instruction
    response = model.generate_content(final_prompt)
    response.resolve()
    print(response)
    if response.parts and hasattr(response.parts[0], 'text'):
        intent_number = int(response.parts[0].text)
        print(intent_number)
        a = intent_number
    return a

def checkRAGIntension(instruction):
    LAB_prompt = "Please help me confirm which intention the content below represents. If there is a fall occurrence, please select intention 1. If it involves taking medication, please select intention 2. If it relates to exercising, please select intention 5. If it concerns sleeping, please select intention 10. Content:"
    
    model = genai.GenerativeModel('gemini-pro')
    final_prompt = LAB_prompt + instruction
    response = model.generate_content(final_prompt)
    response.resolve()
    print(response)
    if response.parts and hasattr(response.parts[0], 'text'):
        intent_text = response.parts[0].text
        match = re.search(r'\d+', intent_text)
        if match:
            intent_number = int(match.group(0))
        else:
            intent_number = 0 
        print(intent_number)
        a = intent_number
    return a


sheet_id = "1JgIbh7V6eAu2B22VRF6gRFmfQHZbu6bDIRw5737nLJ4"

import re
def remove_newline(input_str):
    output_str = re.sub(r'\n\d+', '', input_str)
    output_str = output_str.replace('\n', '')
    output_str = output_str.replace('*', '')
    return output_str
    
def get_Fall_Data(data, instruction):
    LAB_prompt = "I would like to search for something in the following CSV data."
    instruction = instruction
    template = "The meaning of the following CSV data is: when the 'intension' column is 1, it indicates a fall. Below is the CSV data:"
    final_prompt = LAB_prompt + instruction + template + data
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(final_prompt)
    response.resolve() 

    try:
        if response.parts and hasattr(response.parts[0], 'text'):
            result_text = response.parts[0].text
            return remove_newline(result_text)
        else:
            return "No text found"
    except Exception as e:
        print(f"Error processing the response: {e}")
        return "Error processing the response"

def get_Medicine_Data(data, instruction):
    LAB_prompt = "I would like to search for something in the following CSV data."
    instruction = instruction
    template = "The meaning of the following CSV data is: when the 'intension' column is 2, it indicates medication data. Below is the CSV data:"
    final_prompt = LAB_prompt + instruction + template + data
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(final_prompt)
    response.resolve() 

    try:
        if response.parts and hasattr(response.parts[0], 'text'):
            result_text = response.parts[0].text
            return remove_newline(result_text)
        else:
            return "No text found"
    except Exception as e:
        print(f"Error processing the response: {e}")
        return "Error processing the response"

def get_Exercise_Data(data, instruction):
    LAB_prompt = "I would like to search for something in the following CSV data."
    instruction = instruction
    template = "The meaning of the following CSV data is: when the 'intension' column is 5, it indicates excercise time. Below is the CSV data:"
    final_prompt = LAB_prompt + instruction + template + data
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(final_prompt)
    response.resolve() 

    try:
        if response.parts and hasattr(response.parts[0], 'text'):
            result_text = response.parts[0].text
            return remove_newline(result_text)
        else:
            return "No text found"
    except Exception as e:
        print(f"Error processing the response: {e}")
        return "Error processing the response"

def get_FindSTH_Data(data, instruction):
    LAB_prompt = "I would like to search for something in the following CSV data."
    instruction = instruction
    template = "The meaning of the following CSV data is: when the 'intension' column is 9, it indicates I'm looking for something. Below is the CSV data:"
    final_prompt = LAB_prompt + instruction + template + data
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(final_prompt)
    response.resolve() 

    try:
        if response.parts and hasattr(response.parts[0], 'text'):
            result_text = response.parts[0].text
            return remove_newline(result_text)
        else:
            return "No text found"
    except Exception as e:
        print(f"Error processing the response: {e}")
        return "Error processing the response"
    
def get_Sleep_Data(data, instruction):
    LAB_prompt = "I would like to search for something in the following CSV data."
    instruction = instruction
    template = "The meaning of the following CSV data is: when the 'intension' column is 10, it indicates sleeping time. Below is the CSV data:"
    final_prompt = LAB_prompt + instruction + template + data
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(final_prompt)
    response.resolve() 

    try:
        if response.parts and hasattr(response.parts[0], 'text'):
            result_text = response.parts[0].text
            return remove_newline(result_text)
        else:
            return "No text found"
    except Exception as e:
        print(f"Error processing the response: {e}")
        return "Error processing the response"

SPEECH_API_URL = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"
SPEECH_HEADERS = {"Authorization": "Bearer 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"}

TTS_API_URL = "https://api-inference.huggingface.co/models/Nithu/text-to-speech"
TTS_HEADERS = {"Authorization": "Bearer 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"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    audio_data = file.read()
    transcription_text = query_speech(audio_data) 
    if transcription_text:
        return jsonify({'transcription': transcription_text})
    else:
        return jsonify({'error': 'Failed to transcribe audio'}), 500


from PIL import Image
import io

def FindSomething(instruction, img_cache):
    img_cache.seek(0) 
    img = Image.open(img_cache)

    LAB_prompt = "I want to search for something I'm about to tell you about. Please specifically tell me where the item I'm looking for is located in the picture."
    final_prompt = LAB_prompt + instruction
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([final_prompt, img], stream=True)
    response.resolve()

    try:
        if response.parts and hasattr(response.parts[0], 'text'):
            return remove_newline(response.parts[0].text)
        else:
            return "No valid response or text part found"
    except Exception as e:
        print(f"Error processing the response: {e}")
        return "Error processing the response"
    
def startExercise(instruction):
    LAB_prompt = "I want to do some simple indoor exercises for seniors to help prevent degeneration. Please give me instructions on what I should do. Simplify the instructions as much as possible; I just need to perform one complete movement."
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(LAB_prompt)
    response.resolve()

    try:
        if response.parts and hasattr(response.parts[0], 'text'):
            return remove_newline(response.parts[0].text)
        else:
            return "No valid response or text part found"
    except Exception as e:
        print(f"Error processing the response: {e}")
        return "Error processing the response"
    
def exerciseCheck(img_cache, exercise_prompt):
    img_cache.seek(0)
    img = Image.open(img_cache)

    LAB_prompt = "I would like you to help me confirm if the exercise movements of the person in the following photos match my description of the exercise. If the movements are correct, please provide positive encouragement. If not, offer constructive and optimistic suggestions (keep the tone positive)."
    exercise_prompt = exercise_prompt
    final_prompt = LAB_prompt + exercise_prompt
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([final_prompt, img], stream=True)
    response.resolve()

    try:
        if response.parts and hasattr(response.parts[0], 'text'):
            return remove_newline(response.parts[0].text)
        else:
            return "No valid response or text part found"
    except Exception as e:
        print(f"Error processing the response: {e}")
        return "Error processing the response"
    
import re

def normalize_text(input_text):
    input_str = str(input_text)
    if input_str.isdigit():
        return int(input_str)
    numbers = re.findall(r'\d+', input_str)
    if numbers:
        return int(numbers[0])
    return 0 



def query_speech(audio_data):
    response = requests.post(SPEECH_API_URL, headers=SPEECH_HEADERS, data=audio_data)
    response_json = response.json()
    transcription_text = response_json.get('text', '') 
    instruction = transcription_text
    instruction = "Please check the history data, and let me when was the last time I took medicine."
    returnWords = checkInputIntension(instruction)
    returnWords = normalize_text(returnWords)
    print(returnWords)
    # returnWords = 3
    # instruction = "Please let me know when is the last time I took the medicine"

    global global_image_cache
    global exercise_instruction
    global sport

    if returnWords == 1:
        returnWords = FindSomething(instruction, global_image_cache)
        turn_14_on()
        storeFindData()

    if returnWords == 2:
        # returnWords = startExercise(instruction)
        sport = startExercise(instruction)
        # returnWords = exerciseCheck(global_image_cache, a)
        turn_13_on()
        turn_14_on()
        turn_15_on()
        storeExerciseData()
        returnWords = sport
        exercise_instruction = sport
    
    if returnWords == 3:
        RAGIntension = checkRAGIntension(instruction)
        RAGIntension = normalize_text(RAGIntension)
        if RAGIntension == 1:
            data = export_recent_to_csv_string(sheet_id)
            # print(get_Fall_Data(data, instruction))
            returnWords = get_Fall_Data(data, instruction)
            storeRAGData()
        if RAGIntension == 2:
            data = export_recent_to_csv_string(sheet_id)
            # print(get_Fall_Data(data, instruction))
            returnWords = get_Medicine_Data(data, instruction)
            storeRAGData()
        if RAGIntension == 5:
            data = export_recent_to_csv_string(sheet_id)
            # print(get_Fall_Data(data, instruction))
            returnWords = get_Exercise_Data(data, instruction)
            storeRAGData()
        if RAGIntension == 10:
            data = export_recent_to_csv_string(sheet_id)
            # print(get_Fall_Data(data, instruction))
            returnWords = get_Sleep_Data(data, instruction)
            storeRAGData()
    
    if returnWords == 4:
        returnWords = exerciseCheck(global_image_cache, exercise_instruction)
        
    return returnWords

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech_api():

    text = request.json.get("text") 
    print
    if text is None:
        return jsonify({'error': 'Text parameter is missing'})

    audio = query_tts(text)
    if audio is None:
        return jsonify({'error': 'Failed to generate speech'})

    return audio


@app.route('/toggle-light/<int:light_number>/<action>', methods=['POST'])
def toggle_light(light_number, action):
    if action == 'on':
        response = eval(f"turn_{light_number}_on()")
    else:
        response = eval(f"turn_{light_number}_off()")
    return response

@app.route('/store-medicine-data', methods=['POST'])
def store_medicine_data():
    try:
        data = request.json.get('data')

        # Your code for handling data and writing to Google Sheets

        creds = Credentials.from_service_account_file('starry-hawk-419101-b66c43b71517.json', scopes=scope)
        client = gspread.authorize(creds)

        sheet_id = '1JgIbh7V6eAu2B22VRF6gRFmfQHZbu6bDIRw5737nLJ4'
        sheet = client.open_by_key(sheet_id).sheet1
        tz = ZoneInfo("Asia/Taipei")
        today_date = datetime.now(tz).strftime('%Y-%m-%d')
        today_time = datetime.now(tz).strftime('%H:%M:%S')

        intension = 2
        gas_state = 2
        alert_state = 2
        message_state = 2
        lamp_state = 2
        ac_state = 2

        row_data = [today_date, today_time, intension, gas_state, alert_state, message_state, lamp_state, ac_state]

        sheet.append_row(row_data)
        
        return jsonify({'message': 'Data stored successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def query_tts(text):
    payload = {"inputs": text}
    response = requests.post(TTS_API_URL, headers=TTS_HEADERS, json=payload, stream=True)
    try:
        response.raise_for_status()
        return response.content, 200
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None, response.status_code

if __name__ == '__main__':
    app.run(debug=True)