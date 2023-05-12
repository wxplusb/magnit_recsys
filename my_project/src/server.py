import hashlib
import os
from pathlib import Path
import pandas as pd
from loguru import logger
from flask import Flask, render_template, request, redirect, url_for
from .config import *
import numpy as np
import pickle
from surprise import (
    Dataset,
    SVD,
    Reader,
    accuracy
)

# Создаем логгер и отправляем информацию о запуске
# Важно: логгер в Flask написан на logging, а не loguru,
# времени не было их подружить, так что тут можно пересоздать 
# logger из logging
logger.add(LOG_FOLDER + "log.log")
logger.info("Наш запуск")

# Создаем сервер и убираем кодирование ответа
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  


@app.route("/<task>")
def main(task: str):
    """
    Эта функция вызывается при вызове любой страницы, 
    для которой нет отдельной реализации

    Пример отдельной реализации: add_data
    
    Параметры:
    ----------
    task: str
        имя вызываемой страницы, для API сделаем это и заданием для сервера
    """
    return render_template('index.html', task=task)

@app.route("/add_data", methods=['POST'])
def upload_file():
    """
    Страница на которую перебросит форма из main 
    Здесь происходит загрузка файла на сервер
    """
    def allowed_file(filename):
        """ Проверяем допустимо ли расширение загружаемого файла """
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    # Копируем шаблон ответа для сервера и устанавливаем выполняемую задачу
    answer = ANSWER.copy()
    answer['Задача'] = 'add_data'

    # Проверяем наличие файла в запросе
    if 'file' not in request.files:
        answer['Сообщение'] = 'Нет файла'
        return answer
    file = request.files['file']

    # Проверяем что путь к файлу не пуст
    if file.filename == '':
        answer['Сообщение'] = 'Файл не выбран'
        return answer
    
    # Загружаем
    if file and allowed_file(file.filename):
        filename = hashlib.md5(file.filename.encode()).hexdigest() 
        file.save(
            os.path.join(
                UPLOAD_FOLDER, 
                'input.csv'
                )
            )
        # file.save(
        #     os.path.join(
        #         UPLOAD_FOLDER, 
        #         filename + file.filename[file.filename.find('.'):]
        #         )
        #     )
        answer['Сообщение'] = 'Файл успешно загружен!'
        answer['Успех'] = True
        answer['Путь'] = filename
        return answer
    else:
        answer['Сообщение'] = 'Файл не загружен'
        return answer
        
@app.route("/show_data", methods=['GET'])
def show_file():
    """
    Страница выводящая содержимое файла
    """
   
    # Копируем шаблон ответа для сервера и устанавливаем выполняемую задачу
    answer = ANSWER.copy()
    answer['Задача'] = 'show_file'

    # Проверяем, что указано имя файла
    if 'path' not in request.args:
        answer['Сообщение'] = 'Не указан путь файла'
        return answer
    file = request.args.get('path') 
    
    # Проверяем, что указан тип файла
    if 'type' not in request.args:
        answer['Сообщение'] = 'Не указан тип файла'
        return answer
    type = request.args.get('type')

    file_path = os.path.join(UPLOAD_FOLDER, 'output.csv')

    # Проверяем, что файл есть
    if not os.path.exists(file_path):
        answer['Сообщение'] = 'Файл не существует'
        return answer

    answer['Сообщение'] = 'Файл успешно загружен!'
    answer['Успех'] = True
    
    # Приводим данные в нужный вид
    if type == 'csv':
        answer['Данные'] = pd.read_csv(file_path).to_dict()
        return answer
    else:
        answer['Данные'] = 'Не поддерживаемы тип'
        return answer

def recom(x):
    t1 = x.iloc[0]
    top1 = {int(t1.JID):float(t1.Rating)}
    top10 = x['JID'].iloc[:10].to_list()
    return [top1,top10]

@app.route("/start", methods=['POST'])
def start_model():
    logger.info('start_model')
    df = pd.read_csv(os.path.join(
                UPLOAD_FOLDER, 
                'input.csv'
                ))

    logger.info('load input.csv')
    with open('my_project/src/algo.pik', 'rb') as f:
        algo = pickle.load(f)

    logger.info('load algo')

    df = pd.DataFrame(df['UID'],columns=['UID'])
    df_out = df.copy()

    items = pd.DataFrame(np.arange(1,101),columns=['JID'])

    df = df.merge(items,how='cross')
    df['Rating'] = df[['UID', 'JID']].apply(lambda x: algo.predict(x[0], x[1], verbose=False).est,axis = 1)
    logger.info('calc rating')

    df = df.sort_values(['UID','Rating'],ascending=False)

    recs = df.groupby('UID').apply(recom)

    df_out['recommendations'] = recs.reindex(df_out['UID']).reset_index(drop=True)

    df_out.to_csv(os.path.join(
                UPLOAD_FOLDER, 
                'output.csv'
                ),index=False)
    logger.info('save output.csv')
