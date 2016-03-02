import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import math

orig_data = pd.read_csv("network_backup_dataset.csv");
orig_data = pd.DataFrame(orig_data)
'''
'''
#Problem 1: plot all size of backup
workid0 = orig_data[orig_data.WorkFlowID==0]
workid1 = orig_data[orig_data.WorkFlowID==1]
workid2 = orig_data[orig_data.WorkFlowID==2]
workid3 = orig_data[orig_data.WorkFlowID==3]
workid4 = orig_data[orig_data.WorkFlowID==4]

plt.figure()
p0 = plt.subplot(231)
p1 = plt.subplot(232)
p2 = plt.subplot(233)
p3 = plt.subplot(234)
p4 = plt.subplot(235)

plt.sca(p0)
plt.plot(workid0['SizeofBackup'])
p0.set_xlabel('Time')
p0.set_ylabel('Size of Backup')
p0.set_title('Size of Backup--Time(Work-Flow-ID-0)')

plt.sca(p1)
plt.plot(workid1['SizeofBackup'])
p1.set_xlabel('Time')
p1.set_ylabel('Size of Backup')
p1.set_title('Size of Backup--Time(Work-Flow-ID-1)')

plt.sca(p2)
plt.plot(workid2['SizeofBackup'])
p2.set_xlabel('Time')
p2.set_ylabel('Size of Backup')
p2.set_title('Size of Backup--Time(Work-Flow-ID-2)')

plt.sca(p3)
plt.plot(workid3['SizeofBackup'])
p3.set_xlabel('Time')
p3.set_ylabel('Size of Backup')
p3.set_title('Size of Backup--Time(Work-Flow-ID-3)')

plt.sca(p4)
plt.plot(workid4['SizeofBackup'])
p4.set_xlabel('Time')
p4.set_ylabel('Size of Backup')
p4.set_title('Size of Backup--Time(Work-Flow-ID-4)')

plt.show()

#Problem2-Linear Regression
print "Linear Regression"

clf = linear_model.LinearRegression();
cv=cross_validation.KFold(len(orig_data), n_folds=10, shuffle=False)


train = orig_data.loc[:,['WeekNumber','DayofWeek','BackupStartTime','WorkFlowID','FileName','BackupTime']]
target = orig_data.loc[:,['SizeofBackup']]

linear_predict = np.zeros(len(orig_data))
for traincv, testcv in cv:
    print ("Linear Regression Fitting")
    clf.fit(train.iloc[traincv],target.iloc[traincv])
    print ("Linear Regression Predicting")
    linear_predict[testcv] = clf.predict(train.iloc[testcv])
RSS_linear = target.SizeofBackup-linear_predict
RSS_value = mean_squared_error(target.SizeofBackup,linear_predict)**0.5
plt.plot(linear_predict,label='Prediction',color = 'blue')
plt.plot(target,label='Real value',color = 'red')
plt.xlabel('Time')
plt.ylabel('Copy Size')
plt.axis([0,20000,-0.1,1.2])
plt.legend()
plt.title('Copy size versus Time based on Linear Regression')
plt.show()
plt.scatter(linear_predict,RSS_linear,color = 'black')
plt.xlabel('Predicted copy size')
plt.ylabel('Residual')
plt.show()


#Problem3-Random Forest
print "RandomForest"
rf_predict = np.zeros(len(orig_data))
estimator_start = 20
estimator_end = 38
depth_start = 4
depth_end = 10
tune = np.zeros([(estimator_end-estimator_start)/2+1,depth_end-depth_start+1])
for e in range(estimator_start,estimator_end,2):
   for m in range(depth_start,depth_end,1):
      row=(e-estimator_start)/2+1
      col=m-depth_start+1
      tune[0,col] = m
      tune[row,0] = e
      rf = RandomForestRegressor(n_estimators=e, max_depth=m)
      for traincv, testcv in cv:
         rf.fit(train.iloc[traincv],target.iloc[traincv])
         rf_predict[testcv] = rf.predict(train.iloc[testcv])
      RSS_2=target.SizeofBackup-rf_predict
      error = mean_squared_error(target.SizeofBackup,rf_predict)**0.5
      tune[row,col] = error
print (tune)
pos = np.argmin(tune[1:row+1,1:col+1])
e = int(tune[pos/col+1,0])
m = int(tune[0,pos%col+1])
rf = RandomForestRegressor(n_estimators=e, max_depth=m)
for traincv, testcv in cv:
   rf.fit(train.iloc[traincv],target.iloc[traincv])
   rf_predict[testcv] = rf.predict(train.iloc[testcv])
RSS_2=target.SizeofBackup-rf_predict
RSS_2_value=mean_squared_error(target.SizeofBackup,rf_predict)**0.5
plt.plot(target, label='Real value',color='blue')
plt.plot(rf_predict,label='Prediction value',color='red')
plt.xlabel('Time')
plt.ylabel('Copy size')
plt.legend()
plt.title('Copy size versus Time based on Random Forest')
plt.show()
plt.scatter(rf_predict, RSS_2, color='black')
plt.show()


#Problem4-Polynomial
print "Polynomial Regression"
poly_start = 2
poly_end = 10

RSS_poly_1 = np.zeros(poly_end-poly_start)
poly_p = np.zeros(len(orig_data))
for degree in range(poly_start,poly_end,1):
    poly = make_pipeline(PolynomialFeatures(degree), Ridge())
    poly.fit(train, target)
    poly_p = poly.predict(train)
    poly_predict = np.zeros(len(orig_data))
    for i in range(len(orig_data)):
        poly_predict[i] = poly_p[i]
    #plt.plot(target, label="degree %d" % degree, color='blue')
    #plt.plot(poly_predict, label="degree %d" % degree, color='red')
    #plt.show()
    RSS_poly_1[degree-poly_start]=mean_squared_error(target.SizeofBackup,poly_predict)**0.5

plt.plot(range(poly_start,poly_end,1),RSS_poly_1,color='black')
plt.xlabel('Degree')
plt.ylabel('RMSE')
plt.title('Degree versus RMSE for fixed data set')
plt.show()

poly_start = 2
poly_end = 7

RSS_poly_2 = np.zeros(poly_end-poly_start)
poly_p = np.zeros(len(orig_data))
for degree in range(poly_start,poly_end,1):
   poly = make_pipeline(PolynomialFeatures(degree), Ridge())
   for traincv, testcv in cv:
      poly.fit(train.iloc[traincv],target.iloc[traincv])
      poly_p[testcv] = poly.predict(train.iloc[testcv])
      poly_predict = np.zeros(len(orig_data))
      for i in range(len(orig_data)):
          poly_predict[i] = poly_p[i]
   RSS_poly_2[degree-poly_start]=mean_squared_error(target.SizeofBackup,poly_predict)**0.5
plt.plot(range(poly_start,poly_end,1),RSS_poly_2,color='red')
plt.xlabel('Degree')
plt.ylabel('RMSE')
plt.title('Degree versus RMSE for cross validation')
plt.show()

'''
#Problem5-Neural Network Regression
mlp_predict = np.zeros(len(orig_data))
print ("Neural NetWork Regression")
hidden_layer = 100
mlp = MLPRegressor(algorithm='sgd',hidden_layer_sizes = 100)
for traincv,testcv in cv:
    mlp.fit(train.iloc[traincv],target.iloc[traincv])
    mlp_predict[testcv] = mlp.predict(train_iloc[testcv])
RSS_mlp=(target.SizeofBackup-rf_predict).abs()
plt.plot(target, color='blue')
plt.plot(mlp_predict, color='red')
plt.show()
plt.scatter(mlp_predict, RSS_mlp, color='black')
plt.show()
'''

#Problem4-WorkFlowID
train0 = workid0.loc[:,['WeekNumber','DayofWeek','BackupStartTime','WorkFlowID','FileName','BackupTime']]
target0 = workid0.loc[:,['SizeofBackup']]
train1 = workid1.loc[:,['WeekNumber','DayofWeek','BackupStartTime','WorkFlowID','FileName','BackupTime']]
target1 = workid1.loc[:,['SizeofBackup']]
train2 = workid2.loc[:,['WeekNumber','DayofWeek','BackupStartTime','WorkFlowID','FileName','BackupTime']]
target2 = workid2.loc[:,['SizeofBackup']]
train3 = workid3.loc[:,['WeekNumber','DayofWeek','BackupStartTime','WorkFlowID','FileName','BackupTime']]
target3 = workid3.loc[:,['SizeofBackup']]
train4 = workid4.loc[:,['WeekNumber','DayofWeek','BackupStartTime','WorkFlowID','FileName','BackupTime']]
target4 = workid4.loc[:,['SizeofBackup']]

linear_predict0 = np.zeros(len(workid0))
linear_predict1 = np.zeros(len(workid1))
linear_predict2 = np.zeros(len(workid2))
linear_predict3 = np.zeros(len(workid3))
linear_predict4 = np.zeros(len(workid4))

clf0 = linear_model.LinearRegression();
cv0=cross_validation.KFold(len(workid0), n_folds=10, shuffle=False)
clf1 = linear_model.LinearRegression();
cv1=cross_validation.KFold(len(workid1), n_folds=10, shuffle=False)
clf2 = linear_model.LinearRegression();
cv2=cross_validation.KFold(len(workid2), n_folds=10, shuffle=False)
clf3 = linear_model.LinearRegression();
cv3=cross_validation.KFold(len(workid3), n_folds=10, shuffle=False)
clf4 = linear_model.LinearRegression();
cv4=cross_validation.KFold(len(workid4), n_folds=10, shuffle=False)

for traincv0, testcv0 in cv0:
    clf0.fit(train0.iloc[traincv0],target0.iloc[traincv0])
    linear_predict0[testcv0] = clf0.predict(train0.iloc[testcv0])
for traincv1, testcv1 in cv1:
    clf1.fit(train1.iloc[traincv1],target1.iloc[traincv1])
    linear_predict1[testcv1] = clf1.predict(train1.iloc[testcv1])
for traincv2, testcv2 in cv2:
    clf2.fit(train2.iloc[traincv2],target2.iloc[traincv2])
    linear_predict2[testcv2] = clf2.predict(train2.iloc[testcv2])
for traincv3, testcv3 in cv3:
    clf3.fit(train3.iloc[traincv3],target3.iloc[traincv3])
    linear_predict3[testcv3] = clf3.predict(train3.iloc[testcv3])
for traincv4, testcv4 in cv4:
    clf4.fit(train4.iloc[traincv4],target4.iloc[traincv4])
    linear_predict4[testcv4] = clf4.predict(train4.iloc[testcv4])
RSS0 = sum((linear_predict0-target0.SizeofBackup)**2)
RSS1 = sum((linear_predict1-target1.SizeofBackup)**2)
RSS2 = sum((linear_predict2-target2.SizeofBackup)**2)
RSS3 = sum((linear_predict3-target3.SizeofBackup)**2)
RSS4 = sum((linear_predict4-target4.SizeofBackup)**2)
sumRSS = (RSS0+RSS1+RSS2+RSS3+RSS4)/len(orig_data)
RSSfinal = math.sqrt(sumRSS)







