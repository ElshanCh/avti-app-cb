from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoTokenizer, AutoModelForTokenClassification, pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras import  layers
import re 
import numpy as np

class AVTI_ChatBot():

    def __init__(self, model_path: str = r'txt_classification.h5'):

        self.model_path = model_path

        #set up facebook/blenderbot-400M-distill
        self.fbcb_tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        self.fbcb_model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

        #set up xlm-roberta-large-finetuned-conll03-english
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
        self.model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

        #Load self-trained text classification model
        self.cl_model = load_model(self.model_path)

    def get_max_len(self):
        # Get the maxLen of the pretrained dataset
        self.input_layer = next(x for x in self.cl_model.layers[::-1] if isinstance(x, layers.InputLayer)).name
        maxLen = self.cl_model.get_layer(self.input_layer).output.shape[1]
        return maxLen

    def fb_chat(self, question):
        # Facebook Chatbot
        input = self.fbcb_tokenizer(question, return_tensors="pt", truncation=True, padding='max_length')
        res = self.fbcb_model.generate(**input)
        res_decoded = self.fbcb_tokenizer.decode(res[0])
        res_decoded = re.search('<s> (.*)</s>', res_decoded).group(1)
        return res_decoded

    def class_predict(self,X_test_indices):
        return str(np.argmax(self.cl_model.predict(X_test_indices)))

    def replace_name(self,fb_res):
        ner_results = self.nlp(fb_res.upper())            
        if len(ner_results) == 0:
            pass
        else:
            name_entities_lst = []
            for i in ner_results:
                if re.search("PER", i['entity']):
                    name_entities_lst.append(i)
            fb_res = fb_res.replace(fb_res[name_entities_lst[0]['start'] : name_entities_lst[-1]['end']], "AVTI")               
        return fb_res