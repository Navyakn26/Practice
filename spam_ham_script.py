import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

spam_ham_data = pd.read_csv('SMSSpamCollection',sep = '\t',names = ['label','messages'])

cleaned_corpus = []
#ps = PorterStemmer()
lz = WordNetLemmatizer()

for i in range(0,len(spam_ham_data)):
    review = re.sub('[^A-Za-z]', ' ', spam_ham_data['messages'][i])
    review = review.lower()
    review = review.split()
    review = [lz.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    cleaned_corpus.append(review)
    

cv = CountVectorizer(max_features = 2500)
X = cv.fit_transform(cleaned_corpus).toarray()


le = LabelEncoder()
spam_ham_data['label'] = le.fit_transform(spam_ham_data['label'])

y = spam_ham_data['label']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=12,stratify=y)


nb_classifier = MultinomialNB()
nb_classifier.fit(X_train,y_train)

y_pred = nb_classifier.predict(X_test)

confu_matrix = confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)





