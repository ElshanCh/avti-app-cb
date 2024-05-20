import time
import string
import joblib
from pickle import dump, load

# String cleaner
def cleaner(x):
        return [a for a in (''.join([a for a in x if a not in string.punctuation])).lower().split()]

# chat with bot
def AVTI_chatbot(model_path='models/DT_chatbot.pkl'):
    print("\U0001F916  \U0001F9E0  Loading the chatbot..")
    
#     model = load(open(model_path, 'rb'))  #Uncomment if you are loading light model
    model = joblib.load(model_path)  ##Uncomment if you are loading heavy model
    
    print("Note: Enter 'quit' to stop the chatbot.\n")
    while True:
        input_ = input('You: ')
        if input_.lower() == 'quit':
            break
        res = model.predict([input_])[0]
        time.sleep(1)
        print('AVTI-Bot: {:s}'.format(res))
        print()


if __name__ == '__main__':
    AVTI_chatbot()