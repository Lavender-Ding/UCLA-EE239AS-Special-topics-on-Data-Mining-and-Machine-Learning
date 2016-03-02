import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
#import matplotlib.pyplot as pl

print "load data..."

from sklearn.datasets import fetch_20newsgroups
categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                'misc.forsale','soc.religion.christian']
twenty_train = fetch_20newsgroups(subset='train', categories=categories,
                                  shuffle=True, random_state=1)
twenty_test = fetch_20newsgroups(subset='test', categories=categories,
                                 shuffle=True, random_state=1)

#pl.hist(twenty_train.target,bins=range(10),color='g')
#pl.show()

###################tf-idf#####################
print "tf-idf for train data..."
stopWords = text.ENGLISH_STOP_WORDS
vectorizer = CountVectorizer(min_df=1,stop_words = stopWords)
train_data_cnt = vectorizer.fit_transform(twenty_train.data)
train_counts = train_data_cnt.toarray()
tfidf_transformer = TfidfTransformer()
train_data_tfidf = tfidf_transformer.fit_transform(train_counts)
train_features = train_data_tfidf.toarray()
del train_counts

print "tf-idf for test data..."
test_data_cnt = vectorizer.transform(twenty_test.data)
test_counts = test_data_cnt.toarray()
test_data_tfidf = tfidf_transformer.transform(test_counts)
test_features = test_data_tfidf.toarray()
del test_counts

##################LSI#########################
print("Performing dimensionality reduction using LSA")
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
k = 50
svd = TruncatedSVD(k)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
train_selected_features = lsa.fit_transform(train_features)
test_selected_features = lsa.transform(test_features)

###########Multi-class classification#############
print "--------------------Multiclass Classification--------------------"
print "Naive Bayes..."
train_label = twenty_train.target
test_label_ori = twenty_test.target
clf = MultinomialNB()
clf.fit(train_features,train_label)
test_label = clf.predict(test_features)

target_names = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                'misc.forsale','soc.religion.christian']

conMatrix = confusion_matrix(test_label_ori, test_label)
print "confusion matrix is:"
print conMatrix

classReport = classification_report(test_label_ori, test_label, target_names=target_names)
print "classification report is:"
print classReport

accu = metrics.accuracy_score(test_label_ori, test_label)
print "Accuracy:"
print accu

print "multiclass SVM classification..."
print "One VS One..."
clf = OneVsOneClassifier(LinearSVC(random_state=0))
clf.fit(train_features,train_label)
test_label = clf.predict(test_features)

conMatrix = confusion_matrix(test_label_ori, test_label)
print "confusion matrix is:"
print conMatrix

classReport = classification_report(test_label_ori, test_label, target_names=target_names)
print "classification report is:"
print classReport

accu = metrics.accuracy_score(test_label_ori, test_label)
print "Accuracy:"
print accu

print "One VS the rest..."
clf = OneVsRestClassifier(LinearSVC(random_state=0))
clf.fit(train_features,train_label)
test_label = clf.predict(test_features)

conMatrix = confusion_matrix(test_label_ori, test_label)
print "confusion matrix is:"
print conMatrix

classReport = classification_report(test_label_ori, test_label, target_names=target_names)
print "classification report is:"
print classReport

accu = metrics.accuracy_score(test_label_ori, test_label)
print "Accuracy:"
print accu







