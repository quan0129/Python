import pandas as pd
import  matplotlib.pyplot as plt
dataframe = pd.read_csv('Advertising.csv')
x = dataframe.values[:,2]
y = dataframe.values[:,4]
plt.scatter(x,y,marker='o')
plt.show()

def predict(new_radio,weight,bias):
    return weight*new_radio+bias

def cosfunction(x,y,weight,bias):
    n = len(x)
    sum_error = 0
    for i in range(n):
        sum_error += (y[i] -(weight*x[i]+bias))**2
        return sum_error/n

def update_weight(x,y,weight,bias,learing_rate):
    n = len(x)
    weight_temp = 0.0
    bias_temp = 0.0
    for i in range(n):
        weight_temp += 