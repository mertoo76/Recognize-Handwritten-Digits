# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 01:22:07 2017

@author: user
"""

import numpy as np 

sig = lambda t: 1/(1+np.exp(-t))

def load_data():
  from sklearn.datasets import load_digits
  digits = load_digits()
  return digits.images, digits.data, digits.target

def show_image(images, index):
  import matplotlib.pyplot as plt
  plt.figure(1, figsize=(1, 1))
  plt.imshow(images[index], cmap=plt.cm.gray_r, interpolation='nearest')
  plt.show()

images, data, target = load_data()
data = data / 16
#show_image(images, 2)

#ÖZELLİK
#özelliğimiz resimin orta ksımına gelen matris alanının toplamının ortalaması olarak aldık
feature=np.full([1797,1],0.0)


for row in range(1797):
    top = images[row][3][3] + images[row][3][4] + images[row][4][3] + images[row][4][4]
    top = top/4
    top=float(top/16) #max 16 geldiği için 0-1 arasına indirgiyoruz.
    feature[row]=top


    
data=np.append(data, feature, axis=1) #özelliği datanın sonuna yeni bir sütun olarak ekliyoruz

#BİAS
bias=np.full([1797,1],1)
#bias = bias[np.newaxis]
data=np.append(data, bias, axis=1)#bias datanın sonuna yeni bir sütun olarak eklendi.



train_data=data[:int(1797/2)]
test_data=data[int(1797/2):]

'''
layer_1_w=np.random.rand(66,5)-0.5
layer_2_w=np.random.rand(5,5)-0.5
layer_3_w=np.random.rand(5,3)-0.5
layer_4_w=np.random.rand(3,10)-0.5
'''

layer_1_w = np.random.uniform(low=-1, high=1, size=(66,5))
layer_2_w = np.random.uniform(low=-1, high=1, size=(5,5))
layer_3_w = np.random.uniform(low=-1, high=1, size=(5,3))
layer_4_w = np.random.uniform(low=-1, high=1, size=(3,10))

#np.random.uniform(low=-0.5, high=0.5, size=(66,5))

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer() 
label_binarizer.fit(target)
target_output = label_binarizer.transform(target) 

target_output_train=target_output[:int(1797/2)]
target_output_test=target_output[int(1797/2):]
eta = 0.5
for epoch in range(0,1000):
    for x,t in zip(train_data, target_output_train):
        x = x[np.newaxis]
        layer_1_o = sig(np.dot(x, layer_1_w))
        layer_2_o = sig(np.dot(layer_1_o,layer_2_w))
        layer_3_o = sig(np.dot(layer_2_o,layer_3_w))
        result = sig(np.dot(layer_3_o,layer_4_w))
        
        layer_4_delta = ((-(t-result))*((1-result)*result))
        layer_3_delta = layer_4_delta.dot(layer_4_w.T)*((1-layer_3_o)*layer_3_o)
        layer_2_delta = layer_3_delta.dot(layer_3_w.T)*((1-layer_2_o)*layer_2_o)
        layer_1_delta = layer_2_delta.dot(layer_2_w.T)*((1-layer_1_o)*layer_1_o)
        
        
        layer_4_w -= eta * (layer_4_delta.T * layer_3_o).T
        layer_3_w -= eta * (layer_3_delta.T * layer_2_o).T
        layer_2_w -= eta * (layer_2_delta.T * layer_1_o).T
        layer_1_w -= eta * (layer_1_delta.T * x).T
    #x.T.dot(layer_1_delta)

#testlerin yapılması, başarı oranının hesaplanması
succes_ratio=0
result_datas=[]
real_datas=[]
for x,t in zip(test_data, target_output_test):
    x = x[np.newaxis]
    layer_1_o = sig(np.dot(x, layer_1_w))
    layer_2_o = sig(np.dot(layer_1_o,layer_2_w))
    layer_3_o = sig(np.dot(layer_2_o,layer_3_w))
    result = sig(np.dot(layer_3_o,layer_4_w))
    #result_data = label_binarizer.inverse_transform(result)
    result_data=np.argmax(result,axis=1)
    t = t[np.newaxis]
    real_data = label_binarizer.inverse_transform(t)
    result_datas.append(result_data[0])
    real_datas.append(real_data[0])
   

from sklearn.metrics import accuracy_score
succes_ratio=accuracy_score(real_datas, result_datas)

