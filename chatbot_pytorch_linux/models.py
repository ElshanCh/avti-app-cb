from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, pipeline
import numpy as np
from utils import *
import torch
import torch.nn.functional as F
from deep_translator import GoogleTranslator

import re
import calendar
import pickle

class AVTI_ChatBot():

    def __init__(self, model_path: str = r'models/model_classification.pt', ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # load the ner model from disk
        self.nlp = pickle.load(open('models/model_custom_nlp.pkl', 'rb'))
        
        # Classification model
        self.model_path = model_path

        #set up facebook/blenderbot-400M-distill
        self.fbcb_tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        self.fbcb_model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
        self.fbcb_model.tokenizer = self.fbcb_tokenizer
        # Set model to evaluation mode
        self.fbcb_model.eval()
        #time.sleep(10)

        # Quantize model
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.fbcb_model,
            {torch.nn.Linear, torch.nn.LayerNorm},
            dtype=torch.float16,
            inplace=False
        )
        
        #Load self-trained text classification model
        print('starting to load classification')
        self.model = torch.jit.load(self.model_path).to(self.device)
        
        ## load tagger
        #self.tagger = flair_tagger
        

    # Facebook Chatbot
    def fb_chat(self, question):
        quantized_generated_text = self.quantized_model.generate(
        input_ids=self.quantized_model.tokenizer(question, return_tensors="pt")["input_ids"])
        return self.quantized_model.tokenizer.decode(quantized_generated_text[0], skip_special_tokens=True)

    
    # Language Translation model 
    def translate(self,question, from_language, to_language):
        return GoogleTranslator(source=from_language, target=to_language).translate(question)
    
    
    # Classification model prediction
    def predict_single(self,x):    
        x = sentences_to_indices(np.array([x]), self.model.word_to_index, self.model.maxLen)
        # create dataset
        x = torch.tensor(x, dtype=torch.long).to(self.device)

        pred = self.model(x).detach()
        pred = F.softmax(pred,dim=1).to(self.device).numpy()

        pred = pred.argmax(axis=1)

        return pred[0] 

    
    def extract_entity(self,text):    
        from_location = None
        to_location = None
        event_date = None
        event_time = None
        transport = None
        doc = self.nlp(text)
        for ent in doc.ents:
            time_node = False
            #print(ent)
            
            if ent.label_ == "TRANSPORT":
                transport = ent
            if ent.label_ == "LOC-TO":
                to_location = ent
            if ent.label_ == "LOC-FROM":
                from_location = ent
            if ent.label_ == "DATE" :
                if time_node is True:
                    event_date,_ = parse_date(str(ent.text))
                else:
                    event_date,event_time = parse_date(str(ent.text))
            if ent.label_ == "TIME" and event_time is None:
                _,event_time = parse_date(str(ent.text))
                if event_time is not None:
                    time_node = True
        return str(event_date),str(event_time),str(from_location),str(to_location),str(transport)

    
    # Intent sorting  
    def intent_sorting(self, question):
        txt_class = self.predict_single(question)
        data ={}
        #Facebook Chatbot
        if txt_class == 0:
            txt_class = 0
            action = "fb_chatbot"
            category = "no_category"
            res = self.fb_chat(question)
        elif txt_class == 1:
            txt_class = 1
            action = "name"
            category = "no_category"
            if "hello" in question.lower() or "hi" in question.lower():
                res = 'Hello! My name is AVTI. How may I help you'
            else:
                res = 'My name is AVTI. How may I help you'
        elif txt_class == 2:
            txt_class = 2
            action = "initialize_trip"
            category = "agenda"
            res = "I'm starting to build a trip"
            event_date, event_time, from_location, to_location ,transport= self.extract_entity(question)
            data['eventDate'] = event_date
            data['eventTime'] = event_time
            data['fromLocation'] = from_location
            data['toLocation'] = to_location   
            data['transport'] = transport
        elif txt_class == 3:
            txt_class = 3
            action = "create_event"
            category = "agenda"
            res = "New event initialization"
            # Initialize variables to store the date,time, and "from" and "to" locations
            event_date, event_time, from_location, to_location ,transport= self.extract_entity(question)
            data['eventDate'] = event_date
            data['eventTime'] = event_time
            data['fromLocation'] = from_location
            data['toLocation'] = to_location   
            data['transport'] = transport   
        elif txt_class == 4:
            txt_class = 4
            action = "event_modification"
            category = "agenda"
            res = "I'm starting to modify events"    
        elif txt_class == 5:
            txt_class = 5
            action = "transport_preference_modification"
            category = "onboarding"
            res = "I'm starting to modify transport preferences"       
        
        return res,txt_class,action,category,str(data)