import string
import numpy as np

def read_glove_vecs(glove_file_name):
    with open(glove_file_name, 'r',encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        word_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            word_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return word_to_index, index_to_words, word_to_vec_map

# Definitions 
def remove_punct(text):
    translator = str.maketrans("", "", string.punctuation.replace("'", ""))
    return text.translate(translator)


class Preproces_txt():

    def __init__(self,word_to_index):
        
        """
        Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
        The output shape should be such that it can be given to `Embedding()`

        Arguments:
        X -- array of sentences (strings), of shape (m, 1)
        word_to_index -- a dictionary containing the each word mapped to its index
        max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 

        Returns:
        X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
        """
        self.word_to_index = word_to_index
        
    
    # GRADED FUNCTION: sentences_to_indices
    def sentences_to_indices(self, text , max_len):
        
        X = np.array([text])
        
        
        m = X.shape[0]                                   # number of training examples
        
        ### START CODE HERE ###
        # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
        X_indices = np.zeros((m, max_len))
        lst=[]
        for i in range(m):                               # loop over training examples
            
            # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
            sentence_words =(
                                X[i].lower()
                                    .replace("can't", " can not")\
                                    .replace("n't", " not")\
                                    .replace("'s", " is")\
                                    .replace("'ll", " will")\
                                    .replace("'ve", " have")\
                                    .replace("'re", " are")\
                                    .replace("'m", " am")\
                                    .replace("'d", " would")\
                                    .replace("-", " ")\
                                    .replace("\x97", " ")\
                                    .split()
                            )
                
            # Initialize j to 0
            j = 0
            
            # Loop over the words of 
            for w in sentence_words[:17]:
                if remove_punct(w.lower()) in self.word_to_index:
                    # Set the (i,j)th entry of X_indices to the index of the correct word.

                    X_indices[i, j] = self.word_to_index[remove_punct(w.lower())]

                    # Increment j to j + 1
                    j = j + 1
                else:
                    lst.append(w)
                    j = j + 1          
                
        return X_indices