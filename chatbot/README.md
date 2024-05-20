# **ChatBot**

## 📝**Description**
- This repository represents the development of the ***Open-Domain Conversational ChatBot*** based on [facebook/blenderbot-400M-distill](https://huggingface.co/facebook/blenderbot-400M-distill?text=Hey+my+name+is+Mariama%21+How+are+you%3F) with the aim to use it in the application developed by AVTI with the following modifications implemented to adapt it to company goals:<br />

    - Since this ChatBot has been pre-trained on millions of conversations on Facebook, when you ask from Bot its name it can give each time different names. For this reason, there has been developed a ***Sequence Classification model with LSTM Recurrent Neural Network using TensorFlow*** ([model_ClassifyTXT_tf.ipynb](https://github.com/ElshanCh/ChatBot/blob/main/model_ClassifyTXT_tf.ipynb)). <span style="text-decoration:underline">The aim of the Classification model is to classify the questions/requests to the chatbot, whether in the content of each question/request there is an intention to ask the name of the Bot (if yes -> 1, if no -> 0)</span> .
        - In case of 1, the name in the response to the question/request provided by Facebook ChatBot gets replaced by ***AVTI*** using pre-trained *Name Entity Recognition* (NER) model from 🤗Hugging Face [xlm-roberta-large-finetuned-conll03-english](https://huggingface.co/xlm-roberta-large-finetuned-conll03-english)<br />
        For example:<br />
        **Customer question/request:**  *What is your name?*<br />
        **Bot original response:**  *My name is <span style="color:red">Sarah</span>, what is yours? Do you have any siblings?* <br />
        **Bot modified response:**  *My name is <span style="color:green">AVTI</span>, what is yours? Do you have any siblings?*<br />
        - In case of 0, the client receives not modified response of the ChatBot <br /><br />

    - Since the ***facebook/blenderbot-400M-distill*** model is trained on the English language dataset, another modification that has been implemented was making the ChatBot multilingual (adapting to Italian and French languages). For this purpose, the [deep-translator](https://pypi.org/project/deep-translator/#:~:text=It%20is%20100%25%20free%2C%20unlimited,famous%20translators%20in%20this%20tool) library of python gets used. <br />
    So, when the question/request from the client arrives not in English it first gets translated to English, then passes through the Classification model, NER (if it is required), and finally, it gets translated back to the original language before returning to the client.
    

## ⬇️**Install**

This project requires Python and the following Python libraries installed (a full list of required libraries for this project can be found in [requirenments.txt](requirements.txt)):

- NumPy
- TensorFlow
- transformers
- deep_translator

You will also need to have software installed to run and execute a Jupyter Notebook.

