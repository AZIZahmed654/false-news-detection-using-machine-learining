import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

nltk.download('stopwords')
stopwords_set = set(stopwords.words('english')) 


#Data Gtahering

df = pd.read_csv("train.csv")
df = df.dropna() #Handled Missing values by droping those rows

df.reset_index(inplace=True)


lm = WordNetLemmatizer()
corpus = []
for i in range (len(df)):
    review = re.sub('[^a-zA-Z0-9]', ' ', df['title'][i])
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize(x) for x in review if x not in stopwords_set]
    review = " ".join(review)
    corpus.append(review)



tf = TfidfVectorizer()
x = tf.fit_transform(corpus).toarray()
y = df['label']



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 10, stratify = y )
lg=LogisticRegression()
lg.fit(x_train,y_train)
pickle.dump(lg,open('model2.pkl','wb'))


rf = RandomForestClassifier()
rf.fit(x_train, y_train)
pickle.dump(rf,open('model1.pkl','wb'))

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tf, vectorizer_file)


