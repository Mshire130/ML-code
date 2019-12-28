
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__) 

url = 'https://raw.githubusercontent.com/Mshire130/Ml-datasets/master/Concrete_Data_Yeh.csv'
raw_data = pd.read_csv(url) 
#Scanning for any NaNs
raw_data.isnull().any() 

print(raw_data.head())

#Separating train and test dataset, 70-30 split.
train_data = raw_data.sample(frac=0.7, random_state=0)
test_data = raw_data.drop(train_data.index) 

train_data.head() 

train_label = train_data.pop('csMPa')
test_label = test_data.pop('csMPa')

#normalising data
train_stats = train_data.describe()

train_stats = train_stats.transpose()
train_stats


#Fucntion for normalising data
def norm(x):
  return ((x - train_stats['mean'])/train_stats['std'])


#Normalising data
norm_train_data = norm(train_data)
norm_test_data = norm(test_data)

norm_train_data.head() 

#building model 
model = keras.Sequential([
        keras.layers.Dense(64, activation = tf.nn.relu, input_shape = [len(train_data.keys())]), 
        keras.layers.Dense(64, activation = tf.nn.relu),
        keras.layers.Dense(1) ])


model.compile( loss = 'mean_squared_error',
               optimizer = keras.optimizers.RMSprop(0.001),
               metrics = ['mean_squared_error','mean_absolute_error']
          ) 


history = model.fit(norm_train_data, train_label, epochs = 500, validation_split = 0.2)


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.head()) 

#Function for plotting MAE and MSE
def plot_history(history):
  plt.figure()
  plt.plot(hist['epoch'], hist['mean_squared_error'], label = 'Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Val error')
  plt.xlabel('Epoch')
  plt.ylabel('Mean Squared Error')
  plt.ylim([0,400])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Absolute Error')
  plt.plot(hist['epoch'], hist['mean_absolute_error'], label = 'Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Val error')
  plt.ylim([0,20])
  plt.legend()
  
  plt.show()

plot_history(history) 

loss, mse, mae = model.evaluate(norm_test_data, test_label)

print(f'Testing set Mean Abs Error : {mae} MPa')

testPredictions = model.predict(norm_test_data)

plt.figure()
plt.scatter(test_label, testPredictions)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.plot([0,0],[80,80]) 

print("Finished") 

tf.keras.models.save_model(model,"testmodel.h5")