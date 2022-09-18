
import matplotlib
matplotlib.rcParams['font.family'] = 'NanumGothic'  

import matplotlib.pyplot as plt
plt.rc('font', family='NanumBarunGothic')



import os
import sys
print('Python version:', sys.version, '\n')

import platform
print(platform.system())
print(platform.release())

import numpy as np
print('numpy version:', np.__version__, '\n')

import pandas as pd
print('pandas version:', pd.__version__, '\n')

import sklearn
print('scikit-learn version:', sklearn.__version__, '\n')

import tensorflow as tf
print('tf version:', tf.__version__)

import keras
print('keras version:', keras.__version__)

import matplotlib 
print('matplotlib version:', matplotlib.__version__, '\n')

import seaborn as sns 
print('seaborn version:', sns.__version__, '\n')

#  ���۵���̺꿡  work������ �ε��   sample.txt�� ������ ��
DF = pd.read_table('/content/gdrive/MyDrive/WORK/sample.txt', header=0, sep='\s+')
print(DF)

DF_sx = DF.groupby('sx').describe()
print(DF_sx)
print(DF_sx.stack())

t = np.arange(0.0, 2.0, 0.1)
s = 1 + np.sin(2*np.pi*t)
print(t)
print(s)

plt.plot(t,s)
plt.grid(True)
plt.title('�ѱ۵Ǵ��� Ȯ��')
plt.show()

DF = pd.read_table('/content/gdrive/MyDrive/work/sample.txt', header=0, sep='\s+') #, dtype={'sx':str, 'ht':float, 'wt':float})
print(DF)
print(DF.dtypes)

DF_sx = DF.groupby('sx').describe()
print(DF_sx)
print(DF_sx.stack())

plt.plot(DF['wt'], DF['ht'], 'ro')
plt.show()

plt.boxplot([DF[DF['sx']=='F']['wt'], DF[DF['sx']=='M']['wt']])
plt.show()

p = sns.boxplot(x='sx', y='ht', data=DF)
p.set(xlabel='����', ylabel='Ű', title='���ڱ׸�')
plt.show()

