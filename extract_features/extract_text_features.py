import pickle
from nltk.tokenize import word_tokenize
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


class Text_Feature_Extractor:

    def __init__(self, file_path):
        self.file_path = file_path
        self.lemmatizer = WordNetLemmatizer()
        self.ps = PorterStemmer()

    def read_text(self):
        with open(self.file_path, "rb+") as f:
            word_list = pickle.load(f)
        
        return word_list

    def clean_text(self, text_list):
        token_list = []

        for caption in text_list:
            tagged_sentence = nltk.tag.pos_tag(caption.split())
            proper_noun_removed = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS']
            edited_sentence=' '.join(proper_noun_removed)
            
            #remove punctuations
            cleaned_sent = edited_sentence.translate(str.maketrans('', '', string.punctuation))

            #split into words
            words_list = word_tokenize(cleaned_sent)

            #remove stop words
            filtered_list = []
            for word in words_list:
                word = word.lower()
                if (word not in stop_words) and (word.isalpha()):
                    #stemming
                    stemmed_word = self.ps.stem(word)
                    
                    #lemmatization
                    lemmatized_word= self.lemmatizer.lemmatize(stemmed_word)
                    
                    #removing very small words
                    if(len(lemmatized_word)> 5):
                        filtered_list.append(lemmatized_word)

            token_list.append(filtered_list)
        
        return token_list


    def transform_into_embedding(self):
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        

        