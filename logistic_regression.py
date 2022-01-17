#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      lenovo
#
# Created:     22-04-2021
# Copyright:   (c) lenovo 2021
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import csv
import re
from IPython.display import HTML
import random
import seaborn as sns
import nltk
import networkx as nx
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer as PS
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_curve, roc_auc_score

import preprocessor as p
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from lime import lime_tabular

# prepare input data
def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
	X_train_enc = oe.transform(X_train)
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc

#prepare targets
def prepare_targets(y_train, y_test):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc

def select_features(X_train, y_train, X_test):

	fs = SelectKBest(score_func=chi2, k=400)
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)

	return X_train_fs, X_test_fs, fs

def main():
    pass

if __name__ == '__main__':
    main()

##TweetID_file=pd.read_csv('new_training_data.csv', encoding='utf8', header=None, names=['selftext','ANNOTATIONS'])

with open('SDCNL/file1.csv', encoding="utf8", errors="ignore") as f:
    TweetID_file=pd.read_csv(f,  names=['selftext','is_suicide'])

corpus=pd.DataFrame(TweetID_file,columns= ['selftext','is_suicide'])


X_array=list(corpus['selftext'])
Y_array=list(corpus['is_suicide'])


setoftweets=[]

maintain_sentiments_dict=dict()
stopwords_english = stopwords.words('English')
total=len(corpus['selftext'])
porter=PS()


for j in range(0,total):
    each_element=str(corpus['selftext'][j])

    temp_processed_tweet=each_element.split()

    tweets_clean=[]

    for eachword in temp_processed_tweet:
                eachword=porter.stem(eachword)
                if (eachword not in stopwords_english):  # remove stopwords
                    tweets_clean.append(eachword)
    string = ' '.join(tweets_clean)
    setoftweets.append(string)

#newsetofwords=getnewst(setoftweets)
##
X_array=list(setoftweets)


X_train, X_test, y_train, y_test = train_test_split(X_array, Y_array, test_size = 0.25, random_state=42)

tfidf = TfidfVectorizer(ngram_range=(1, 2))
#tfidf = CountVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
#tfidf = HVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
X_train = tfidf.fit_transform(X_train)


df=pd.DataFrame(
    X_train.todense(),
    columns=tfidf.get_feature_names()


)

#print(tfidf.get_feature_names())
X_test=tfidf.transform(X_test)

#X_train_enc, X_test_enc = X_train, X_test
##### prepare output data
####y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
####
####
##### feature selection
####X_train, X_test, fs = select_features(X_train, y_train, X_test)
##
##
logreg = LogisticRegression(C=1)
logreg.fit(X_train, y_train)

print ("Accuracy is %s" % ( accuracy_score(y_test, logreg.predict(X_test))))

Y_predict=logreg.predict(X_test)

get_confusion_matrix=confusion_matrix(y_test, Y_predict)
print(get_confusion_matrix)

tn, fp, fn, tp=get_confusion_matrix.ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

Precision = tp/(tp+fp)
print("Precision {:0.2f}".format(Precision))

#Recall
Recall = tp/(tp+fn)
print("Recall {:0.2f}".format(Recall))

f1 = (2*Precision*Recall)/(Precision + Recall)
print("F1 Score {:0.2f}".format(f1))

Specificity = tn/(tn+fp)
print("Specificity {:0.2f}".format(Specificity))

Sensitivity = tp/(tp+fn)
print("Sensitivity {:0.2f}".format(Sensitivity))

aur=roc_auc_score(y_test,Y_predict)
print(get_confusion_matrix)

##labels = [0, 1]
##class_names = ['Depression', 'Suicide']

### Plot confusion matrix in a beautiful manner
##fig = plt.figure(figsize=(16, 14))
##ax= plt.subplot()
##sns.heatmap(get_confusion_matrix, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
### labels, title and ticks
##ax.set_xlabel('Predicted', fontsize=20)
##ax.xaxis.set_label_position('bottom')
##plt.xticks(rotation=90)
##ax.xaxis.set_ticklabels(class_names, fontsize = 10)
##ax.xaxis.tick_bottom()
##
##ax.set_ylabel('Actual', fontsize=20)
##ax.yaxis.set_ticklabels(class_names, fontsize = 10)
##plt.yticks(rotation=0)
##
##plt.title('Depression v/s Suicide: Confusion Matrix for Validation', fontsize=20)
##
##plt.savefig('ConMat_validate_LR.pdf')
#plt.show()
##
##target=[0,1,2,3,4,5]
##print(classification_report(y_test, Y_predict))

with open('SDCNL/file2.csv', encoding="utf8", errors="ignore") as f_s:
    TweetID_file_s=pd.read_csv(f_s, header=None, names=['selftext','is_suicide'])

corpus_s=pd.DataFrame(TweetID_file_s,columns= ['selftext','is_suicide'])


new_corpus=corpus_s #.iloc[1:, :]


X_test_s=list(new_corpus['selftext'])
#X_test_s=X_test_s[1:]
y_test_s=list(new_corpus['is_suicide'])
#y_test_s=y_test_s[1:]


setoftweets=[]
maintain_sentiments_dict=dict()
total=len(new_corpus['selftext'])



for j in range(1,total):
    each_element=str(new_corpus['selftext'][j])

    temp_processed_tweet=each_element.split()

    tweets_clean=[]

    for eachword in temp_processed_tweet:
                eachword=porter.stem(eachword)
                if (eachword not in stopwords_english):  # remove stopwords
                    tweets_clean.append(eachword)
    string = ' '.join(tweets_clean)

    setoftweets.append(string)

#newsetofwords=getnewst(setoftweets)
##
X_test_s=list(setoftweets)


X_test_s=tfidf.transform(X_test_s)

y_test_s=y_test_s[1:]
Y_predict_s=logreg.predict(X_test_s)
Y_predict_s=list(Y_predict_s)
print ("Accuracy is %s" % ( accuracy_score(y_test_s, Y_predict_s )))



get_confusion_matrix=confusion_matrix(y_test_s, Y_predict_s)
print(get_confusion_matrix)

tn, fp, fn, tp=get_confusion_matrix.ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

Precision = tp/(tp+fp)
print("Precision {:0.2f}".format(Precision))

#Recall
Recall = tp/(tp+fn)
print("Recall {:0.2f}".format(Recall))

f1 = (2*Precision*Recall)/(Precision + Recall)
print("F1 Score {:0.2f}".format(f1))

Specificity = tn/(tn+fp)
print("Specificity {:0.2f}".format(Specificity))

Sensitivity = tp/(tp+fn)
print("Sensitivity {:0.2f}".format(Sensitivity))

aur=roc_auc_score(y_test_s,Y_predict_s)
print(get_confusion_matrix)

labels = [0, 1]
class_names = ['depression','suicide']

# Plot confusion matrix in a beautiful manner
##fig1 = plt.figure(figsize=(16, 14))
##ax= plt.subplot()
##sns.heatmap(get_confusion_matrix, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
### labels, title and ticks
##ax.set_xlabel('Predicted ', fontsize=20)
##ax.xaxis.set_label_position('bottom')
##plt.xticks(rotation=90)
##ax.xaxis.set_ticklabels(class_names, fontsize = 10)
##ax.xaxis.tick_bottom()
##
##ax.set_ylabel('Actual ', fontsize=20)
##ax.yaxis.set_ticklabels(class_names, fontsize = 10)
##plt.yticks(rotation=0)
##
##plt.title('Depression v/s Suicide: Confusion Matrix for Testing', fontsize=20)
##
##plt.savefig('ConMat_test_LR.pdf')
###plt.show()

####EXPLAINABILITY
##print(type(X_train))
##df=pd.DataFrame(
##    X_train.todense(),
##    columns=tfidf.get_feature_names()
##
##)
##
##explainer = lime_tabular.LimeTabularExplainer(X_train, mode="classification", feature_names= tfidf.get_feature_names())
##print(explainer)
##
###idx = random.randint(1, len(X_test_s))
##explanation = explainer.explain_instance(X_test_s[5], logreg.predict_proba,
##                                         num_features=len(tfidf.get_feature_names()))
##
##
##with plt.style.context("ggplot"):
##    explanation.as_pyplot_figure()
##
##with plt.style.context("ggplot"):
##    fig = plt.figure(figsize=(8,5))
##    plt.barh(range(len(logreg.coef_)), logreg.coef_, color=["red" if coef<0 else "green" for coef in logreg.coef_])
##    plt.yticks(range(len(logreg.coef_)), btfidf.get_feature_names());
##    plt.title("Weights")
##
##print(explanation.as_list())
##print(explanation.as_map())
##
##html_data = explanation.as_html()
##HTML(data=html_data)
##
###explanation.save_to_file(".html")