import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.datasets import fetch_20newsgroups
import math
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *

print ("load data...")
category = ['alt.atheism','comp.graphics','comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware','comp.sys.mac.hardware',
              'comp.windows.x','misc.forsale','rec.autos','rec.motorcycles',
              'rec.sport.baseball','rec.sport.hockey','sci.crypt','sci.electronics',
              'sci.med','sci.space','soc.religion.christian','talk.politics.guns',
              'talk.politics.mideast','talk.politics.misc','talk.religion.misc']
twenty_train = fetch_20newsgroups(subset='train', categories=category,
                                  shuffle=True, random_state=1)
print ("tf-idf")

stopWords = text.ENGLISH_STOP_WORDS
vectorizer = CountVectorizer(min_df=1,stop_words = stopWords,analyzer='word')
tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()
traindata = []
for i in range(len(twenty_train.data)):
    twenty_train.data[i] = tokenizer.tokenize(twenty_train.data[i])
    twenty_train.data[i] = [stemmer.stem(plural) for plural in twenty_train.data[i]]
    traindata = traindata + twenty_train.data[i]
train_data_cnt = vectorizer.fit_transform(traindata)
feature_name = vectorizer.get_feature_names()
cont_feature = np.zeros([len(feature_name),20])
resulteval = np.zeros([len(feature_name),20])
max_freq = np.zeros(20);
num_exist = np.zeros(len(feature_name))

def cal_tfxidf(i):
    global feature_name, cont_feature, category, stopWords
    traini = fetch_20newsgroups(subset='train', categories=[category[i]], shuffle=True, random_state=1)
    v_tmp = CountVectorizer(min_df=1,stop_words = stopWords,analyzer='word')
    vector = v_tmp.fit_transform(traini.data)
    varr = vector.toarray()
    print(len(varr))
    for j in range(len(feature_name)):
        flag = v_tmp.vocabulary_.get(feature_name[j])
        if (flag!=None):
            sumf = sum(varr[:,flag])
            cont_feature[j][i] = sumf
    return

for n in range(20):
    cal_tfxidf(n)

for p in range(20):
    max_freq[p] = max(cont_feature[:,p])

for n in range(len(feature_name)):
    tmp = cont_feature[n,:]
    num_exist[n] = len(tmp[tmp!=0])

for n in range(len(feature_name)):
    for m in range(20):
        resulteval[n][m] = (0.5 + 0.5 * cont_feature[n][m] / max_freq[m]) * math.log(20 / num_exist[n])

indx = np.array([3,4,6,15])
for n in range(4):
    argre = resulteval[:,indx[n]].argsort()
    print ("==============================")
    print (category[indx[n]])
    for m in range(10):
        print (feature_name[argre[-1*m-1]])
    print ("==============================")

