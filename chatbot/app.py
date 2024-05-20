import os
#import psycopg2
import pg8000.native
from dotenv import load_dotenv
from flask import Flask, request
import datetime
from utils import Preproces_txt, read_glove_vecs
from models import AVTI_ChatBot
from deep_translator import GoogleTranslator


#Load glove.6B.50d.txt
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')

#Instanciate the Bot
avti_cb = AVTI_ChatBot()
maxLen = avti_cb.get_max_len()

CREATE_MESSAGES_TABLE = (
    "CREATE TABLE IF NOT EXISTS cb.messages (user_id SERIAL4, language VARCHAR, user_message VARCHAR, cb_response VARCHAR, created_on TIMESTAMP)"
)

INSERT_IN_MESSAGES =(
    "INSERT INTO cb.messages (user_id,language,user_message,cb_response,created_on) VALUES (%s,%s,%s,%s,%s)"
) 

load_dotenv()

app = Flask(__name__)
# url = os.getenv("DATABASE_URL")
# connection = psycopg2.connect(url)
# connection = psycopg2.connect("dbname='avti_cb_test'  user='avti_cb_t' password='wW2QiaR6AUk6JmweR3h'")

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
dbname = os.getenv("dbname")
port = os.getenv("port")
connection = pg8000.native.Connection(host=host, user=user, password=password, dbname=dbname, port =port)


@app.post("/api/message")
def get_message():

    data = request.get_json()
    language = data['language'] 
    question = data['message']
    user_id = data['user_id']
   
    #Get date
    ct = datetime.datetime.now()
    #Pre-process data
    prep = Preproces_txt(word_to_index)

    if language == 'en':
        X_test_indices = prep.sentences_to_indices(question, maxLen)
        #Facebook Chatbot
        fb_res = avti_cb.fb_chat(question)
        #Txt Classifier
        txt_class = avti_cb.class_predict(X_test_indices)
        if txt_class == '1':
            fb_res = avti_cb.replace_name(fb_res)

    else:    
        #Translate to english    
        to_en = GoogleTranslator(source=language, target='en')
        from_en = GoogleTranslator(source='en', target=language)
        question_tr = to_en.translate(question)

        X_test_indices = prep.sentences_to_indices(question_tr, maxLen)
        # Facebook Chatbot
        fb_res = avti_cb.fb_chat(question_tr)
        # Txt Classifier
        txt_class = avti_cb.class_predict(X_test_indices)
        if txt_class == '1':
            fb_res = avti_cb.replace_name(fb_res)
        #Translate back to origin language
        fb_res = from_en.translate(fb_res)
        
    #Sink data to DataBase
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_MESSAGES_TABLE)
            cursor.execute(INSERT_IN_MESSAGES, (user_id, language, question, fb_res, ct))

    return {"user_id" :user_id, "language":language, "user_message" : question, "cb_response" : fb_res, "created_on": ct}