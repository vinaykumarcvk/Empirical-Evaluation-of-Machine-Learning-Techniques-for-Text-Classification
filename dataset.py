import re
import os
import string
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
import time
start_time = time.time()

newsgroups_train = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
X, y = newsgroups_train.data, newsgroups_train.target

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def preprocess(text):
    # Remove punctuations and convert to lower case
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize the text
    words = text.split()
    # Remove stop words and lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def processModel(classifier, label):
    start_time = time.time()
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', classifier),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print('\n',label)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))
    # print execution time
    print('Execution time: ', (time.time() - start_time), 'seconds')
    # save the printed data to a csv file
    # create a directory named label in the same directory as this file if it doesn't exist
    if not os.path.exists(label):
        os.makedirs(label)
    # make the path as the current working directory
    os.chdir(label)
    
    
    with open(label+'.csv', 'a') as f:
        f.write('\n'+label)
        f.write('\nAccuracy: '+str(accuracy_score(y_test, y_pred)))
        f.write('\nExecution time: '+str((time.time() - start_time))+' seconds')
        f.write('\nClassification Report:\n'+str(classification_report(y_test, y_pred)))
    # plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for '+label)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(label+'.png')
    plt.show()
    # plot the ROC curve
    y_pred_proba = pipeline.predict_proba(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(20):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_proba[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for '+label)
    plt.legend(loc="lower right")
    plt.savefig(label+'_ROC.png')
    plt.show()
    
#check if the data is already preprocessed
try:
    X = np.load('preprocessed_data.npy')
except:
    X = [preprocess(text) for text in X]
    #save the preprocessed data
    np.save('preprocessed_data.npy', X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# processModel(LogisticRegression(), 'Logistic Regression')
# processModel(DecisionTreeClassifier(), 'Decision Tree')
# processModel(RandomForestClassifier(), 'Random Forest')
# processModel(KNeighborsClassifier(n_neighbors=5), 'KNN(5 Neighbors)')
# processModel(KNeighborsClassifier(n_neighbors=9), 'KNN(9 Neighbors)')
processModel(MultinomialNB(), 'Multinomial Naive Bayes')
# processModel(BernoulliNB(), 'Bernoulli Naive Bayes')
# processModel(SVC(), 'SVM')