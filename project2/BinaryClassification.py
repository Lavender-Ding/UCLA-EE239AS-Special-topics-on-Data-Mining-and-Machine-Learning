import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *
import matplotlib.pyplot as pl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3

print "load data..."

from sklearn.datasets import fetch_20newsgroups
categories = [ 'comp.graphics','comp.os.ms-windows.misc',
               'comp.sys.ibm.pc.hardware','comp.sys.mac.hardware',
               'rec.autos','rec.motorcycles',
               'rec.sport.baseball','rec.sport.hockey']
twenty_train = fetch_20newsgroups(subset='train', categories=categories,
                                  shuffle=True, random_state=1)
twenty_test = fetch_20newsgroups(subset='test', categories=categories,
                                 shuffle=True, random_state=1)

pl.hist(twenty_train.target,bins=range(10),color='g')
pl.xlabel('class')
pl.ylabel('number of files')
pl.title('Histogram')
pl.show()

###################tf-idf#####################
print "tf-idf for train data..."
stopWords = text.ENGLISH_STOP_WORDS
tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()
for i in range(len(twenty_train.data)):
    twenty_train.data[i] = tokenizer.tokenize(twenty_train.data[i])
    twenty_train.data[i] = [stemmer.stem(plural) for plural in twenty_train.data[i]]
    traindata = twenty_train.data[i][0]
    for j in range(len(twenty_train.data[i])):
        traindata = traindata + ' ' + twenty_train.data[i][j]
    twenty_train.data[i] = traindata
vectorizer = CountVectorizer(min_df=1,stop_words = stopWords,analyzer='word')
train_data_cnt = vectorizer.fit_transform(twenty_train.data)
train_counts = train_data_cnt.toarray()
tfidf_transformer = TfidfTransformer()
train_data_tfidf = tfidf_transformer.fit_transform(train_counts)
train_features = train_data_tfidf.toarray()
del train_counts

print "tf-idf for test data..."
for i in range(len(twenty_test.data)):
    twenty_test.data[i] = tokenizer.tokenize(twenty_test.data[i])
    twenty_test.data[i] = [stemmer.stem(plural) for plural in twenty_test.data[i]]
    testdata = twenty_test.data[i][0]
    for j in range(len(twenty_test.data[i])):
        testdata = testdata + ' ' + twenty_test.data[i][j]
    twenty_test.data[i] = testdata
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

###############linearSVM######################
print "linear SVM..."
from sklearn import svm
clf = svm.LinearSVC()
train_label = np.zeros(len(twenty_train.target))
for i in range(0,len(twenty_train.target)):
    if twenty_train.target[i] < 4:
        train_label[i] = 0
    else:
        train_label[i] = 1
clf.fit(train_selected_features, train_label)
test_label = clf.predict(test_selected_features)
test_label_ori = np.zeros(len(twenty_test.target))
for i in range(0,len(twenty_test.target)):
    if twenty_test.target[i] < 4:
        test_label_ori[i] = 0
    else:
        test_label_ori[i] = 1

from sklearn import metrics
pred = np.dot(test_selected_features, clf.coef_.T)
fpr1, tpr1, thresholds1 = metrics.roc_curve(test_label_ori, pred, pos_label=0)
plt1.plot(fpr1[1000:2000],tpr1[1000:2000],label='SVM', color='blue')
plt1.legend(loc = 'upper left')
plt1.xlabel('FPR')
plt1.ylabel('TPR')
plt1.title('ROC curve for SVM')
plt1.show()

accu = metrics.accuracy_score(test_label_ori, test_label)
print "Accuracy:"
print accu

from sklearn.metrics import confusion_matrix
conMatrix = confusion_matrix(test_label_ori, test_label)
print "confusion matrix is:"
print conMatrix

from sklearn.metrics import classification_report
target_names = ['Computer Technology', 'Recreational Activity groups']
classReport = classification_report(test_label_ori, test_label, target_names=target_names)
print "classification report is:"
print classReport

###############soft-margin-SVM##################
print "soft margin SVM..."
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
k = np.array(range(-3,4,1), dtype=np.float64)
c_range = pow(10,k)
i = 0
cv=cross_validation.KFold(len(train_selected_features), n_folds=5, shuffle=False)
cross_pred = np.zeros(len(train_label))
accuracy = np.zeros(7)
for c in c_range:
    clf = svm.SVC(kernel = 'linear', C = c)
    for traincv, testcv in cv:
        clf.fit(train_selected_features[traincv],train_label[traincv])
        cross_pred[testcv] = clf.predict(train_selected_features[testcv])
    accuracy[i] = metrics.accuracy_score(train_label, cross_pred)
    i = i + 1
max_index = np.argmax(accuracy)
c_opt = c_range[max_index]
clf = svm.SVC(kernel = 'linear', C = c_opt)
clf.fit(train_selected_features,train_label)
test_label = clf.predict(test_selected_features)
accu = metrics.accuracy_score(test_label_ori, test_label)
print "Accuracy:"
print accu

conMatrix = confusion_matrix(test_label_ori, test_label)
print "confusion matrix is:"
print conMatrix

classReport = classification_report(test_label_ori, test_label, target_names=target_names)
print "classification report is:"
print classReport

#################Naive Bayes#####################
print "Naive Bayes..."
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(train_features,train_label)
test_label = clf.predict(test_features)
test_score = clf.predict_proba(test_features)
fpr2, tpr2, thresholds2 = metrics.roc_curve(test_label_ori, test_score[:,1], pos_label=0)
plt2.plot(fpr2[1000:2000],tpr2[1000:2000],label='Naive Bayes', color='blue')
plt2.legend(loc = 'upper left')
plt2.xlabel('FPR')
plt2.ylabel('TPR')
plt2.title('ROC curve for Naive Bayes')
plt2.show()

conMatrix = confusion_matrix(test_label_ori, test_label)
print "confusion matrix is:"
print conMatrix

classReport = classification_report(test_label_ori, test_label, target_names=target_names)
print "classification report is:"
print classReport

accu = metrics.accuracy_score(test_label_ori, test_label)
print "Accuracy:"
print accu

##############logistic Regression################
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(train_selected_features,train_label)
test_label = clf.predict(test_selected_features)
test_score_log = clf.predict_proba(test_selected_features)
fpr3, tpr3, thresholds3 = metrics.roc_curve(test_label_ori, test_score_log[:,1], pos_label=0)
plt3.plot(fpr3[1000:2000],tpr3[1000:2000],label='Logistic', color='blue')
plt3.legend(loc = 'upper left')
plt3.xlabel('FPR')
plt3.ylabel('TPR')
plt3.title('ROC curve for Logistic Regression')
plt3.show()

plt.plot(fpr1[1000:2000],tpr1[1000:2000],label='SVM', color='blue')
plt.plot(fpr2[1000:2000],tpr2[1000:2000],label='Naive Bayes', color='red')
plt.plot(fpr3[1000:2000],tpr3[1000:2000],label='Logistic', color='green')
plt.legend(loc = 'upper left')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()

conMatrix = confusion_matrix(test_label_ori, test_label)
print "confusion matrix is:"
print conMatrix

classReport = classification_report(test_label_ori, test_label, target_names=target_names)
print "classification report is:"
print classReport

accu = metrics.accuracy_score(test_label_ori, test_label)
print "Accuracy:"
print accu

