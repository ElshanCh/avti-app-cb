import os
from dotenv import load_dotenv
from flask import Flask, request
import datetime
from models import AVTI_ChatBot
#import psycopg2
import traceback
from flask_cors import cross_origin
import mysql.connector




CREATE_MESSAGES_TABLE = (
    "CREATE TABLE IF NOT EXISTS avti_cb_test.messages (user_id VARCHAR(36), language VARCHAR(255), user_message VARCHAR(255), cb_response VARCHAR(255), error_txt VARCHAR(255), intent INT, created_on TIMESTAMP)"
)

INSERT_IN_MESSAGES =(
    "INSERT INTO avti_cb_test.messages (user_id, `language`, user_message, cb_response, intent, created_on) VALUES (%s, %s, %s, %s, %s, %s)"
) 

INSERT_IN_MESSAGES_ERROR =(
    "INSERT INTO avti_cb_test.messages (user_id, `language`, user_message, error_txt, created_on) VALUES (%s, %s, %s, %s, %s)"
)


#Instanciate the Bot
avti_cb = AVTI_ChatBot()
load_dotenv('./env')
app = Flask(__name__)

# CORS(app, resources = {r'/api/*':{'origins':["http://localhost:3000/", "http://cillene.lunalabs.it:15303/chatbot-open/", "https://dev.avti.lunalabs.it/"]}})

host = os.getenv("host")
user = os.getenv("user")
if user is None:
    user = "avti_cb_t"
password = os.getenv("password")
dbname = os.getenv("dbname")
port = os.getenv("port")


@app.route("/api/message", methods=['GET', 'POST'])
@cross_origin()
def get_message():
    data = request.get_json()
    language = data['language'] 
    question = data['message']
    user_id = data['userId']
    ct = datetime.datetime.now()    

    try:        
        err = 'Null'
        success = True

        if language == 'en':
            res,txt_class,action,category,data = avti_cb.intent_sorting(question)            
        else:    
            #Translate to english    
            question_tr = avti_cb.translate(question=question, from_language=language, to_language='en')
            res,txt_class,action,category,data = avti_cb.intent_sorting(question_tr)
            #Translate back to origin language
            res = avti_cb.translate(question=res, from_language='en', to_language=language)


        #connection = psycopg2.connect(host=host, user=user, password=password, dbname=dbname, port =port)
        # establish connection to MySQL server
        connection = mysql.connector.connect(user=user, password=password,
                                    host=host, database=dbname, port=port)
        #print(user_id, language, question, res,txt_class, ct)
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(CREATE_MESSAGES_TABLE)
                cursor.execute(INSERT_IN_MESSAGES, (user_id, language, question, res,txt_class, ct))
                connection.commit()
        cursor.close()
        connection.close()

    except Exception as e: 
        err = str(e)
        success = False
        res = "NULL"
        action = "NULL"
        data = "NULL"
        txt_class = "NULL"
        category = "NULL"

        print(traceback.format_exc())
        
        #connection = psycopg2.connect(host=host, user=user, password=password, dbname=dbname, port =port)
        # establish connection to MySQL server
        connection = mysql.connector.connect(user=user, password=password,
                                    host=host, database=dbname, port=port)
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(CREATE_MESSAGES_TABLE)
                cursor.execute(INSERT_IN_MESSAGES_ERROR, (user_id, language, question, err, ct))
                connection.commit()
        cursor.close()
        connection.close()

    return {"userID": user_id, "language": language, "userMessage": question, "cbResponse": res,"success": success, "errorMessage":err,"data": data, "intent": txt_class, "action" : action, "category": category, "createdOn": ct}