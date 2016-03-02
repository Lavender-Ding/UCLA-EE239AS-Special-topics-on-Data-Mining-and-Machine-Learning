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
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

print "Read Data"
in_data = "housing_data.csv"
indata = pd.read_csv(in_data,header=None)
indata = pd.DataFrame(indata)

print "Linear Regression"
clf = linear_model.LinearRegression()
cv=cross_validation.KFold(len(indata), n_folds=10, shuffle=False)
results=[]

data = indata.iloc[:,0:13]
label = indata[13]
pre_label_1 = np.zeros(len(data))

for traincv, testcv in cv:
   clf.fit(data.iloc[traincv],label.iloc[traincv])
   pre_label_1[testcv] = clf.predict(data.iloc[testcv])
RSS_1=label-pre_label_1
plt.plot(label.index, label, label='true label', color='blue')
plt.plot(label.index, pre_label_1, label='predict label', color='red')
plt.xlabel('Sample Number')
plt.ylabel('Label')
plt.title('Linear Regression Prediction')
plt.legend(loc = 'upper left')
plt.show()
plt.scatter(pre_label_1, RSS_1, color='black')
plt.xlabel('Fitted Value')
plt.ylabel('Residual')
plt.title('Residuals vs Fitted values')
plt.show()

'''
print "RandomForest"
pre_label_2 = np.zeros(len(data))
estimator_start = 18
estimator_end = 40
depth_start = 4
depth_end = 12
tune = np.zeros([(estimator_end-estimator_start)/2+1,depth_end-depth_start+1])
for e in range(estimator_start,estimator_end,2):
   for m in range(depth_start,depth_end,1):
      row=(e-estimator_start)/2+1
      col=m-depth_start+1
      tune[0,col] = m
      tune[row,0] = e
      rf = RandomForestRegressor(n_estimators=e, max_depth=m)
      for traincv, testcv in cv:
         rf.fit(data.iloc[traincv],label.iloc[traincv])
         pre_label_2[testcv] = rf.predict(data.iloc[testcv])
      RSS_2=(label-pre_label_2)**2
      error = RSS_2.mean()
      tune[row,col] = error
print tune
pos = np.argmin(tune[1:row+1,1:col+1])
e = int(tune[pos/col+1,0])
m = int(tune[0,pos%col+1])
rf = RandomForestRegressor(n_estimators=e, max_depth=m)
for traincv, testcv in cv:
   rf.fit(data.iloc[traincv],label.iloc[traincv])
   pre_label_2[testcv] = rf.predict(data.iloc[testcv])
RSS_2=(label-pre_label_2)**2
plt.plot(label.index, label, color='blue')
plt.plot(label.index, pre_label_2, color='red')
plt.show()
plt.scatter(pre_label_2, RSS_2, color='black')
plt.show()
'''

print "Polynomial Regression"
poly_start = 2
poly_end = 10
RSS_poly_1 = np.zeros(poly_end-poly_start)
pre_label_3 = np.zeros(len(data))
for degree in range(poly_start,poly_end,1):
   poly = make_pipeline(PolynomialFeatures(degree), Ridge())
   poly.fit(data, label)
   pre_label_3 = poly.predict(data)
   plt.plot(label.index, label, label='true label', color='blue')
   plt.plot(label.index, pre_label_3, label='predict label', color='red')
   plt.xlabel('Sample Number')
   plt.ylabel('Label')
   plt.title("Polynomial Regression: degree = %d" % degree)
   plt.legend(loc = 'upper left')
   plt.show()
   RSS_poly_1[degree-poly_start]=mean_squared_error(label, pre_label_3)**0.5

plt.plot(range(poly_start,poly_end,1),RSS_poly_1,color='black')
plt.xlabel('Degree')
plt.ylabel('RMSE')
plt.title('RMSE vs Degree using fixed train and test data')
plt.show()

poly_start = 2
poly_end = 6

RSS_poly_2 = np.zeros(poly_end-poly_start)
pre_label_4 = np.zeros(len(data))
for degree in range(poly_start,poly_end,1):
   poly = make_pipeline(PolynomialFeatures(degree), Ridge())
   for traincv, testcv in cv:
      poly.fit(data.iloc[traincv],label.iloc[traincv])
      pre_label_4[testcv] = poly.predict(data.iloc[testcv])
   plt.plot(label.index, label, label='true label', color='blue')
   plt.plot(label.index, pre_label_4, label='predict label', color='red')
   plt.xlabel('Sample Number')
   plt.ylabel('Label')
   plt.title("Polynomial Regression with 10-fold: degree = %d" % degree)
   plt.legend(loc = 'upper left')
   plt.show()
   RSS_poly_2[degree-poly_start]=mean_squared_error(label, pre_label_4)**0.5

plt.plot(range(poly_start,poly_end,1),RSS_poly_2,color='red')
plt.xlabel('Degree')
plt.ylabel('RMSE')
plt.title('RMSE vs Degree using 10-fold')
plt.show()


print "Ridge Regression"
RSS_ridge = np.zeros(3)
pre_label_ridge = np.zeros(len(data))
i=0;
for a in [0.1,0.01,0.001]:
   clf_rig = linear_model.Ridge(alpha = a)
   for traincv, testcv in cv:
      clf_rig.fit(data.iloc[traincv],label.iloc[traincv])
      pre_label_ridge[testcv] = clf_rig.predict(data.iloc[testcv])
   RSS_ridge[i]=mean_squared_error(label, pre_label_ridge)**0.5
   i = i + 1
print RSS_ridge

RSS_ridge_lasso = np.zeros(3)
pre_label_ridge_lasso = np.zeros(len(data))
i=0;
for a in [0.1,0.01,0.001]:
   clf_rig = linear_model.Lasso(alpha = a)
   for traincv, testcv in cv:
      clf_rig.fit(data.iloc[traincv],label.iloc[traincv])
      pre_label_ridge_lasso[testcv] = clf_rig.predict(data.iloc[testcv])
   RSS_ridge_lasso[i]=mean_squared_error(label, pre_label_ridge_lasso)**0.5
   i = i + 1
print RSS_ridge_lasso



