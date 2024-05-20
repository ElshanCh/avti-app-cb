import re
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from datetime import datetime, timedelta,time
import re
import calendar

# Preprocessing the data
contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
def clean_text(x):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', x)
    return text
def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x
def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re
contractions, contractions_re = _get_contractions(contraction_dict)
def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)


# GRADED FUNCTION: sentences_to_indices
def sentences_to_indices(X, word_to_index, max_len):

    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()`.     
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    m = X.shape[0]    
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))
    lst=[]
    for i in range(m):  # loop over training examples
       # print(X[i])
        # Convert the ith training sentence in lower case and clean, and then split is into words(with word_tokenize). You should get a list of words.
        sentence =clean_text(clean_numbers(replace_contractions(X[i].lower())))
        sentence_words = word_tokenize(sentence)                      
        # Initialize j to 0
        j = 0
        # Loop over the words of        
        for w in sentence_words:
            if j < 22:
                if w in word_to_index:
                    # Set the (i,j)th entry of X_indices to the index of the correct word.
                    X_indices[i, j] = word_to_index[w]
                    # Increment j to j + 1
                    j = j + 1
                else:
                    lst.append(w)
                    j = j + 1
            else: break   
    return X_indices


def read_glove_vecs(glove_file):
    with open(glove_file, 'r',encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def find_transport(sentence):
    words_list = ["train", "subway", "tram", "bus", "shared bike", "shared scooter", "shared motorbike", "shared car", "bicycle", "bike", "walk", "scooter", "motorbike",  "car", "shared car", "taxi", "airplane", "ferry"]
    return next((word for word in words_list if word in sentence), None)

## FUNCTIONS TAKEN FROM https://www.kaggle.com/gmhost/gru-capsule
def pretrained_embedding_layer(word_to_index,word_to_vec_map):    
    embeddings_index = word_to_vec_map
    word_index = word_to_index    
    emb_mean,emb_std = -0.005838499,0.48782197
    embed_size = embeddings_index["cucumber"].shape[0]
    nb_words = len(word_index)+1
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
    return embedding_matrix, nb_words, embed_size


def word_to_number(text):
    num_dict = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
        "seventy": 70, "eighty": 80, "ninety": 90
    }
    
    words = text.lower().split()
    result = 0
    sub_total = 0
    for word in words:
        if word == "and":
            result += sub_total
            sub_total = 0
            continue
        num = num_dict.get(word, None)
        if num is not None:
            sub_total += num
        elif word == "hundred":
            sub_total *= 100
        elif word == "thousand":
            result += sub_total * 1000
            sub_total = 0
        else:
            return None # invalid input
    return result + sub_total

def parse_date(input_str):        
    now = datetime.now()
    date = None
    t = None
    # Match time formats using regex
    time_match = re.search(r'(?P<hour>\d{1,2})(?:[:|.](?P<minute>\d{2}))? ?(?P<am_pm>am|pm)?', input_str)
    if time_match is not None and sum(value is not None for value in time_match.groupdict().values()) >= 2:
        # Extract matched values from regex match
        hour = int(time_match.group('hour'))
        minute = int(time_match.group('minute') or 0)
        am_pm = time_match.group('am_pm')
        if hour > 12:
            pass
        # Convert 12-hour format to 24-hour format if necessary
        if am_pm and am_pm.lower() == 'pm' and hour != 12:
            hour += 12
        #print(hour)
        
        # Create time object
        if hour < 24:
            if minute:
                t = time(hour=hour, minute=minute)
            else:
                t = time(hour=hour)
                #print(t)
        
    # Parsing time
    time_regex = re.search(r'\b(\d+)\s*[hrs|hours|hr|h|hs]+\b', input_str, re.IGNORECASE)
    if time_regex:
        hr = int(time_regex.group(1))
        t = now + timedelta(hours=hr)
        date = t.date() if t.date() > now.date() else now.date()
        t = t.strftime('%H:%M:%S')
        #print(date, t)

    # Today
    if date is None and re.search(r'\btoday\b', input_str, re.IGNORECASE):
        date = now.date()
        #print(date)

    # Tomorrow
    if date is None and re.search(r'\btomorrow\b', input_str, re.IGNORECASE):
        date = (now + timedelta(days=1)).date()
        #print(date)

    # Day after tomorrow
    if date is None and re.search(r'\bthe day after tomorrow\b', input_str, re.IGNORECASE):
        date = (now + timedelta(days=2)).date()
        #print(date)

    # Weekday
    if date is None:
        weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for i, weekday in enumerate(weekdays):
            if re.search(r'\b' + weekday + r'\b', input_str, re.IGNORECASE):
                day = i - now.weekday()
                day += 7 if day <= 0 else 0
                date = (now + timedelta(days=day)).date()
                #print(date)
                break

    # Date with "in x days" format
    days_regex = re.search(r'\bin?\s*(.*)\s+days?\b', input_str, re.IGNORECASE)
    if date is None and days_regex:
        try:
            days = int(days_regex.group(1))
        except:
            days = days_regex.group(1)
            days=word_to_number(days)
        date = (now + timedelta(days=days)).date()
        #print(date)
    # Date with "in x weeks" format
    weeks_regex = re.search(r'\bin?\s*(.*)\s+weeks?\b', input_str, re.IGNORECASE)    
    if date is None and weeks_regex:
        try:
            weeks = int(weeks_regex.group(1))
            date = (now + timedelta(weeks=weeks)).date()
        except:
            weeks_str = weeks_regex.group(1)
            weeks = word_to_number(weeks_str)
        date = (now + timedelta(weeks=weeks)).date()
    
    # Date with "in x months" format
    months_regex = re.search(r'\bin?\s*(.*)\s+months?\b', input_str, re.IGNORECASE)
    if date is None and months_regex:
        try:
            months = int(months_regex.group(1))
        except:
            months = months_regex.group(1)
            months=word_to_number(months)


        year = now.year + (now.month + months - 1) // 12
        month = (now.month + months - 1) % 12 + 1
        day = min(now.day, calendar.monthrange(year, month)[1])
        date = datetime(year, month, day).date()
        #print(date)
    
    # Date with "in x years" format
    years_regex =re.search(r'\bin?\s*(.*)\s+years?\b', input_str, re.IGNORECASE)
    if date is None and years_regex:
        try:
            years = int(years_regex.group(1))
        except:
            years = years_regex.group(1)
            years=word_to_number(years)
        date = (now + timedelta(days=years*365)).date()
        #print(date)
    
    # Date with "next week" format
    if date is None and re.search(r'\bnext week\b', input_str, re.IGNORECASE):
        date = (now + timedelta(weeks=1)).date()
        #print(date)

    # Date with "next month" format
    if date is None and re.search(r'\bnext month\b', input_str, re.IGNORECASE):
        year = now.year + (now.month + 1) // 12
        month = (now.month + 1) % 12 
        day = min(now.day, calendar.monthrange(year, month)[1])
        date = datetime(year, month, day).date()
        #print(date)
    # Date with "next year" format
    if date is None and re.search(r'\bnext year\b', input_str, re.IGNORECASE):
        year = now.year + 1
        month = now.month
        day = min(now.day, calendar.monthrange(year, month)[1])
        date = datetime(year, month, day).date()
        #print(date)

    # Date with "april, 30", "april 30", "april30","30 Apr","30 apr", "30th April", "april 30th", "30th of the April" format
    date_pattern = re.compile(r'(?P<day1>\d*)[\s|,|th]*(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)[\s|,]*(?P<day2>\d*)', re.IGNORECASE)
    if date is None and re.search(date_pattern, input_str):
        match = date_pattern.search(input_str)
        month_str = match.group('month')
        if match.group('day1'):
            day_str = match.group('day1')            
        elif match.group('day2'):
            day_str = match.group('day2')
        else:
            day_str = None
        month_dict = {
            'jan': 'January','feb': 'February','mar': 'March','apr': 'April','may': 'May','jun': 'June','jul': 'July','aug': 'August','sep': 'September','oct': 'October','nov': 'November','dec': 'December'
        }

        if month_str.lower()[:3] in month_dict:
            month_str = month_dict[month_str.lower()[:3]]

        month = datetime.strptime(month_str, '%B').month
        if day_str is None:
            day_str = '1'
        date = datetime(now.year, month, int(day_str)).date()
        #print(date) 
    
    return date, t