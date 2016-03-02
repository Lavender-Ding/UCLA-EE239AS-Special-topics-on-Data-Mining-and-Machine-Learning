from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error



data = pd.read_csv('network_backup_dataset.csv')
train = data.loc[:,['WeekNumber','DayofWeek','BackupStartTime','WorkFlowID','FileName','BackupTime']]
target = data.loc[:,['SizeofBackup']]
mlp = MLPRegressor(algorithm='sgd', hidden_layer_sizes=150,
                   max_iter=200, shuffle=False, random_state=1)

mlp.fit(train, target)
prediction = mlp.predict(train)

plt.plot(prediction,label='Prediction',color='red')
plt.plot(target,label='Real Data',color='blue')
plt.title('Copy Size versus Time based on Neural Network Regression')
plt.xlabel('Time')
plt.ylabel('Copy Size')
plt.legend()
plt.show()

rmse = mean_squared_error(target.SizeofBackup,prediction)**0.5
print (rmse)