from flask import Flask, request
from models import AVTI_chatbot_DT
from utils import cleaner
# import json
from transformers import pipeline, Conversation, AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

@app.route("/AVTI_chatbot/DTClassifier/<input_string>",  methods=['GET', 'POST'])
def chat_w_bot(input_string):
    #extract interest string from json
    # input_json = json.loads(input_json)
    # print(input_json)
    # input_sent = input_json['user_sent']
    output_str = AVTI_chatbot_DT(input_sent=input_string)
    return output_str

@app.route("/AVTI_chatbot/HF_FB_BlenderBot/<input_string>",  methods=['GET', 'POST'])
def chat_w_bot_HF_BB(input_string):
    chatbot_pipeline = pipeline("conversational",model='facebook/blenderbot-400M-distill')
    conv = Conversation(input_string)
    cb_output = chatbot_pipeline(conv)
    for n, res in enumerate(cb_output.iter_texts()):
        if n==1:
            output_str = res[1]
    return output_str

if __name__ == '__main__':
    app.run(debug=True)