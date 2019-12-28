# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
#print(tf.__version__) 

from random import seed
from random import randint

#Loading and printing model summary 

new_model = tf.keras.models.load_model("testmodel.h5")      

print(new_model.summary()) 


#create stats object 
class stats:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std 
    def normalize(self, value, mean, std):
        self.normalized = ((value - mean) / std)
#Set variables as stats object, define mean and std of each one
cement = stats(281.346602,104.505455)
slag = stats(71.850902,85.812598)
flyash = stats(55.238419,63.853538)
water = stats(182.108044,21.215630)
superplasticizer = stats(6.091262,5.998052)
coarseaggregate = stats(974.500000,76.926675)
fineaggregate = stats(771.734535,80.189816) 
age = stats(45.313454,62.168311)

#Ask questions, Gather information regarding value of inputs

cement.value = int(input("How much cement content (Kg in a m3 mixture)? min:102 max:540 \n" ))
slag.value = int(input("How much slag content (Kg in a m3 mixture)? min:0 max:359 \n" ))
flyash.value = int(input("How much flyash content (Kg in a m3 mixture)? min:0 max:200 \n" ))
water.value = int(input("How much water content (Kg in a m3 mixture)? min:122 max:247 \n" ))
superplasticizer.value = int(input("How much superplasticizer content (Kg in a m3 mixture)? min:0 max:32\n" ))
coarseaggregate.value = int(input("How much coarseaggregate content (Kg in a m3 mixture)? min:801 max:1145\n" ))
fineaggregate.value = int(input("How much fineaggregate content (Kg in a m3 mixture)? min:594 max:992\n" ))
age.value = int(input("How old is the sample (days)? min:1 max:365\n" ))


#Normalizing input value of variables
cement.normalize(cement.value,cement.mean,cement.std)
slag.normalize(slag.value,slag.mean,slag.std)
flyash.normalize(flyash.value,flyash.mean,flyash.std)
water.normalize(water.value,water.mean,water.std)
superplasticizer.normalize(superplasticizer.value,superplasticizer.mean, superplasticizer.std)
coarseaggregate.normalize(coarseaggregate.value,coarseaggregate.mean, coarseaggregate.std)
fineaggregate.normalize(fineaggregate.value,fineaggregate.mean,fineaggregate.std)
age.normalize(age.value,age.mean,age.std) 

#Putting normalized inputs into Dataframe
inputData = [[cement.normalized,slag.normalized,flyash.normalized,water.normalized,superplasticizer.normalized,coarseaggregate.normalized,fineaggregate.normalized,age.normalized]]
PredictionDF = pd.DataFrame(inputData, columns = ['cement','slag','flyash','water','superplasticizer','coarseaggregate','fineaggregate','age'])

print(PredictionDF) 



#Running data through model & printing out the prediction 
#Run 200, 100, 0, 150, 0, 900, 800, 28
concreteStrength = float("{0:.2f}".format(new_model.predict(PredictionDF)[0][0]))
print(f"\nThe strength of your sample is: {concreteStrength} MPa")


#Improvements 1 - Add the +- Error of the predicted strength to the end print 
#Improvements 2 - Set limits to the data thats being input, making sure that the value entered is between the limits mentioned