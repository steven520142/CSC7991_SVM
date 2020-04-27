import sys
import pandas as pd
import numpy as np
import nltk
import pickle
import json
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

review = sys.argv[1]
print(review)
with open('Tfidf_vect.pickle', 'rb') as f:
    Tfidf_vect = pickle.load(f)
with open('SVM.pickle', 'rb') as f:
    SVM = pickle.load(f)

review = str(review)
review = review.lower()
review = word_tokenize(review)
# Declaring Empty List to store the words that follow the rules for this step
Final_words = []
# Initializing WordNetLemmatizer()
word_Lemmatized = WordNetLemmatizer()
# pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
for word, tag in pos_tag(review):
    # Below condition is to check for Stop words and consider only alphabets
    if word not in stopwords.words('english') and word.isalpha():
        word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
        Final_words.append(word_Final)
Final_review = []
Final_review.insert(0, str(Final_words)) 

review_Tfidf = Tfidf_vect.transform(Final_review)
review_predictions_SVM = SVM.predict(review_Tfidf)
print(review_predictions_SVM)
if review_predictions_SVM:
    json_str =json.dumps('Positive');
    print(json_str);
else:
    json_str =json.dumps('Negative');
    print(json_str);