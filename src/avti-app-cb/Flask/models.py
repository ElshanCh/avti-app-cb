import time
import string
import joblib
from utils import cleaner

# chat with bot
def AVTI_chatbot_DT(input_sent: str, model_path: str = r'C:\Users\Davide\Documents\DataScience\AVTI\models\DT_chatbot.pkl'):
    print("\U0001F916  \U0001F9E0  Loading the chatbot..")
    model = joblib.load(model_path)
    
    print("Input sentence: {:s}".format(input_sent))
    res = model.predict([input_sent])[0]
    print('AVTI-Bot: {:s}'.format(res))
    return res