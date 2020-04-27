import pandas as pd
import numpy as np
import nltk
import pickle
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

#Set Random seed
np.random.seed(500)

# Add the Data using pandas
Corpus = pd.read_csv(r"Amazon Reviews Split.csv",encoding='latin1')
IMDB = pd.read_csv(r"IMDB Dataset.csv")
print("Corpus loaded")

# Step - 1: Data Pre-processing - This will help in getting better results through the classification algorithms

# Step - 1a : Remove blank rows if any.
print("Removing blank rows")
#Corpus['Summary'].dropna(inplace=True)
IMDB['review'].dropna(inplace=True)

# Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
print("Changing all the text to lower case")
Corpus['text'] = [str(entry) for entry in Corpus['text']]
Corpus['text'] = [entry.lower() for entry in Corpus['text']]
IMDB['review'] = [entry.lower() for entry in IMDB['review']]

# Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
print("Proccesing Tokenization")
Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]
print("Corpus Done")
IMDB['review']= [word_tokenize(entry) for entry in IMDB['review']]
print("IMDB Done")

# Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
print("Removing Stop words")

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


for index,entry in enumerate(Corpus['text']):
    # Declaring Empty List to store the words that follow the rules for this step
    print(index)
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Corpus.loc[index,'text_final'] = str(Final_words)
print("Corpus Done")    
    
for index,entry in enumerate(IMDB['review']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    IMDB.loc[index,'review_final'] = str(Final_words)
print("IMDB Done")

#print(Corpus['text_final'].head())

# Step - 2: Split the model into Train and Test Data set
print("Spliting the model into Train and Test Data set")
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['label'],test_size=0.3)
#IMDB_Train_X, IMDB_Test_X, IMDB_Train_Y, IMDB_Test_Y = model_selection.train_test_split(IMDB['review_final'],IMDB['sentiment'],test_size=1)
IMDB_Test_X = IMDB['review_final']
IMDB_Test_Y = IMDB['sentiment']
# Step - 3: Label encode the target variable  - This is done to transform Categorical data of string type in the data set into numerical values
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
IMDB_Test_Y = Encoder.fit_transform(IMDB_Test_Y)

# Step - 4: Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comaprison to the corpus
print("Vectorize the words by using TF-IDF Vectorizer")
Tfidf_vect = TfidfVectorizer(max_features=100)
Tfidf_vect.fit(Corpus['text_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
IMDB_Test_X_Tfidf = Tfidf_vect.transform(IMDB_Test_X)


# Step - 5: Now we can run different algorithms to classify out data check for accuracy
# fit the training dataset on the classifier (C=0.5)
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
IMDB_predictions_SVM = SVM.predict(IMDB_Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score (C=0.5) -> ",accuracy_score(predictions_SVM, Test_Y)*100)
print("SVM Accuracy Score (C=0.5, IMDB) -> ",accuracy_score(IMDB_predictions_SVM, IMDB_Test_Y)*100)
# --------------------------------------------------------------------------------
with open('SVM.pickle', 'wb') as f:
    pickle.dump(SVM, f)
with open('Tfidf_vect.pickle', 'wb') as f:
    pickle.dump(Tfidf_vect, f)
