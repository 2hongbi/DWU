#!/usr/bin/env python
# coding: utf-8

# <style>
#     div.cell{
#         width:120%;
#         margin-left:1%;
#         margin-right:auto;
#     }
# </style>
# 
# # Adult Income : 이상치포착모형 1
# 
# * 데이터 처리
#     * adult.csv: 원자료
#     * adult4ml.csv:  ML용자료 
# * AdultUCI (n=48842명, d+1=15), SAS:ZDMLCENS, R, Python
# 
# <!--
# * Association Rule 적용시 이산화함
# * [R 예제: logistic](https://rpubs.com/H_Zhu/235617)
# * [R 예제2: logit, tree, rf](https://rstudio-pubs-static.s3.amazonaws.com/538563_85cb2b4cd06b4dc48d33de73fa97a297.html)
# * [R arules](https://www.kaggle.com/code/mariolatajs/data-mining-adultuci-dataset/notebook)
# * [Python 예제. 처리설명](https://aniket-mishra.github.io/projects/Adult-UCI-Dataset-Analysis.html)
# * 이상치제거 분석결과 
#     * 고려한 모형들의 이상치판단이 매우 다름
#     * 이상치로 판단하여 제거하기보다 모두 다 사용하여 분석모형을 만드는 것이 좋을 듯. 이상치 영향을 많이 받는 모형분석시 유의할 것
# -->    
# 
# 
# 원변수|type     |변수 설명              |전처리. DF:ML용 자료, DF2:arules용 자료 
# ------|:--------|:----------------------|:-----------------------
# age, agegrp 나이|num|right-skewed. (15,25,45,65,100)로 cut,('Y','M'.'S','O')로 레이블|`cut(age, breaks=c(15,25,45,65,100)),labels=c('Yng','Mid','Senior','Old'))`
# workclass 근로계층(wrkcls), wrkclsgrp|nom|Federal-gov(공무원),Local-gov(지방공무원), Never-worked(미취업),Private(개인), Self-emp-inc(자영1),Self-emp-not-inc(자영2),State-gov(주공무원), Without-pay(봉사)|Gov, Gov, Without-pay, Private, Self-emp, Self-emp, Gov, Without-pay
# fnlwgt표집가중치|num|Final sampling weight right-skewed. 제거|
# education 교육수준 (edu)|ordinal|Preschool, 1st-4th, 5th-6th, 7th-8th, 9th, 10th, 11th, 12th,HS-grad, Prof-school, Assoc-acdm, Assoc-voc, Some-college, Bachelors , Masters , Doctorate|1,1,1,1,1,1,1,1,2,5,3,3,3,4,5,6
# education-num교육연수(eduyr)|num|제거|
# marital-status혼인(marital)|nom|Divorced, Married-AF-spouse, Married-civ-spouse, Married-spouse-absent, Never-married, Separated, Widowed| Divorced, Married, Married, NotMarried, NotMarried, Separated, Widowed
# occupation 직업(occ)|nom|Adm-clerical, Armed-Forces, Craft-repair, Exec-managerial, Farming-fishing, Handlers-cleaners, Machine-op-inspct, Other-service, Priv-house-serv, Prof-specialty, Protective-serv, Sales, Tech-support, Transport-moving.|그대로 사용
# relationship관계(rel)|nom|Husband, Not-in-family, Other-relative, Own-child, Unmarried, Wife, 제거|
# race 인종|nom|Amer-Indian-Eskimo, Asian-Pac-Islander, Black, Other, White|Other, Other, Black, Other, White
# sex 성별|nom|Female, Male|
# capital-gain자본소득(capgain)|num|(-Inf,0,median(x>0), Inf)|`cut(capgain,breaks=c(-Inf,0,median(capgain[capgain>0]), inf), labels=c('None','Low','High'))`
# capital-loss 자본손실(caploss)|num|(-Inf,0,median(x>0), Inf) |`cut(caploss, breaks=c(-Inf,0,median(caploss[caploss>0]),Inf), labels=c('None','Low','High'))`
# hours-per-week 주당근로시간(hr)|num|high-kurtosis (0,25,40,60,168) |`cut(hr,breaks=c(0,25,40,60,168)),labels=c('PT','FT','OT','Workaholic'))`
# native-country 출신국(country)|nom|Cambodia, Canada, China, Columbia, Cuba, Dominican-Republic, Ecuador, El-Salvador, England, France, Germany, Greece, Guatemala, Haiti, Holand-Netherlands, Honduras, Hong, Hungary, India, Iran, Ireland, Italy, Jamaica, Japan, Laos, Mexico, Nicaragua, Outlying-US(Guam-USVI-etc), Peru, Philippines, Poland, Portugal, Puerto-Rico, Scotland, South, Taiwan, Thailand, Trinadad&Tobago, United-States, Vietnam, and Yugoslavia| US, UKCA, EU, AsiaN, AsiaS, AmerS
# income (y)|binary|small(연간소득 \\$ 50,000미만), large (연간소득 \\$50,000이상)|
# 
# * 지도학습용 로짓 회귀분석 자료 (Target=income)
# * 타겟 income의 분포 : small 24720 (50.6%), large 7841(16.1%), NA 16281(33.3%)|||
# & arules에서 연관성분석 적용 (http://hope.simons-rock.edu/~cthatcher/math231_spring2012.html)|||
# 
# <!--
# 
# ##  [arules 처리예제](https://rdrr.io/cran/arules/man/Adult.html)
# 
# * numx : 
#    * fnlwgt, education-num, relationship 은 제거
#    * age는 사용자지정 4단계 순서형
#    * hours-per-week는 사용자지정 4단계 순서형
#    * capital-gain: 사용자지정으로 3단계 순서형
#    * captial-loss: 사용자지정으로 3단계 순서형
# * catx는 모두 그대로 사용
# 
# 
# ```
# library(tidyverse)
# adult <- read_csv('D:/WS2022B/dataset/adult.csv') 
# # 문자를 모두 factor로 변경
# adult <- adult %>% mutate_if(is.character, as.factor)
# sapply(adult, class)
# 
# # 이름변경
# names(adult) <- c('age', 'wrkcls', 'fnlwgt', 'edu', 'eduyr', 'marital', 'occ', 'rel', 'race', 'sex',
#  'capgain', 'caploss', 'hr', 'country', 'y')
# 
# 
# # 변수제거
# adult <- adult %>% dplyr::select(-fnlwgt, -`eduyr`)
# 
# 
# adult %>% 
#  mutate(
#    age = ordered( 
#            cut(age, c(15, 25, 45, 65, 100)), 
#            labels = c("Young", "Middle-aged", "Senior", "Old")
#          ),
#    hr = ordered(
#            cut(hr, c(0, 25, 40, 60, 168)),
#            labels = c("PartTime", "FullTime", "OverTime", "Workaholic")
#          ),
#    capgain = ordered(
#            cut(capgain, c(-Inf, 0, median(adult$capgain[adult$capgain>0]), Inf)),
#            labels = c("None", "Low", "High")
#          ),
#    caploss = ordered(
#            cut(caploss, c(-Inf, 0, median(adult$caploss[adult$caploss>0]), Inf)),
#            labels = c("None", "Low", "High")
#          ))
# 
# ```
# 
# 
# * age: 왼쪽 쏠림(right-skewed). 사용자정의 (15, 25, 45, 65, 100)로 cut후 정수화 
# 
# 
# ```
# adult$age <- ordered(('Young','Middle-aged','Senior','Old'), 
# cut(adult$age,breaks=c(15,25,45,65,100)),labels=c('Young','Middle-aged','Senior','Old'))`
# 
# * `mlxtend`: ML Extensions 라이브러리. 연관성분석 함수가 포함되어 있음
#     * `from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth, fpmax`
# 
# 
# ```
# ! pip install mlxtend
# ```
# -->

# # 자료읽기
# 
# ## 다운로드/CSV 저장
# * `sklearn.datasets.fetch_openml`: 
#     * UCI에서 직접 다운로드. Bunch 또는 DF로 다운 가능
#     * Bunch{data(X), target(y), DESCR} 

# In[1]:


import pandas as pd
import numpy as np
# from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# CSV 읽기
# sep=',',  header='infer 또는 0', index_col=None(변수이름 또는 번호), 
# na_values=NA로 인식할 값 추가지정. NA, NaN, N/A, n/a, NaN, nan, NULL, null은 모두 NaN으로 처리됨
# encoding='utf-8'
adultml = pd.read_csv('D:/WS2022B/dataset/adult4ml.csv')  
adultml.head()


# In[3]:


adultml.shape


# ## 자료정리
# 
# * 데이터셋 이름: 
#     * 원본: DF0, TR0, TS0, VL0
#     * 이상치제거 : DF1, TR1, TS1, VL1
#     * 작업원본: DF, TR, TS, VL pipeline 적용된 자료 XX1을 복사한 후 사용
#     * 변환: DF, TRros, TRrus, TRsmt 
# 
# 
# ###  변수이름정리

# In[4]:


# 원본 파일 
DF0 = adultml.copy()
DF0.info()  # numeric, object(문자)로 구성됨 


# In[5]:


# fnlwgt, eduyr 제거  
DF0.drop(['fnlwgt', 'eduyr'], axis=1, inplace=True)


# In[6]:


objname = DF0.select_dtypes('object')


# In[7]:


# 문자변수 일괄 category화.
objname = DF0.select_dtypes('object').columns
for vname in objname:
    DF0[vname] = DF0[vname].astype('category') 
DF0.info()   


# In[8]:


DF0['edugrp'] = pd.Categorical(DF0['edugrp'], ordered=True)
DF0['edugrp']


# In[9]:


# DF['y'] = DF['y'].map({'<=50K': 0, '>50K': 1}) 
DF0.head()


# # 이상치제거 
# * DF: DF0에서 이상치를 제거한 데이터
# * 수치형 특성만 이용하여 판단
# * 이상치포착모형(이상치비율 5%로 지정) 4개의 결과 모두 이상치로 판단된 샘플만 제거
#     * `oneclassSVM`은 사전에 이상치비율을 지정할 수 없어서 튜닝해야 함.
#     * [nu in oneclassSVM](https://github.com/scikit-learn/scikit-learn/issues/12249): 이상치탐지시 nu는 이상치비율와 유사, 특이치탐지시 nu는 FPR(False Positive Rate)이므로 0.05 권장

# In[10]:


from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

DFX0 = DF0.select_dtypes(np.number).copy()  # A value is trying to be set on a copy of a slice from a DataFrame. 방지하기위해 deep copy
DFX0.head()


# In[11]:


#  결측확인
DFX0.isnull().sum()


# ## one-class SVM
# * `sklearn.svm.OneClassSVM(kernel='rbf', gamma='scale', nu=0.5, ...)`
# * `nu`:  포착할 이상치의 비율 대신 사용. 오염비율로 사용 (0, 1]. 모형오차의 상한인 동시에 지지벡터비율의 하한.  

# In[12]:


# OneClassSVM 
seed = 1234
Eonesvm = svm.OneClassSVM(nu=0.02,  kernel='rbf') #, gamma=30)  # nu값이 클수록, gamma값이 작을수록 이상치가 많아짐 (확실하지 않음)
Eonesvm.fit(DFX0)


# In[13]:


# 정상치 점수: 점수가 높을수록 inlier, 점수가 낮을수록 outlier
scr_onesvm = Eonesvm.score_samples(DFX0)


# In[14]:


fig, ax = plt.subplots(2,1)
ax[0].hist(scr_onesvm);
ax[1].boxplot(scr_onesvm, vert=False);


# In[15]:


cutoff = np.quantile(scr_onesvm, 0.02)
isin, nx = np.unique(scr_onesvm>cutoff, return_counts=True)
cutoff, isin, nx, nx/len(scr_onesvm)


# In[16]:


# 정상여부예측값{-1:outlier, 1:inliner}
ihonesvm = Eonesvm.predict(DFX0)
isin, nx = np.unique(ihonesvm, return_counts=True)
isin, nx, nx/len(ihonesvm)


# In[17]:


DFX0.iloc[ihonesvm<0]


# ## EllipticEnvelope
# * `sklearn.covariance.EllipticEnvelope(contamination=0.1, ...)`
# * 다변량정규분포를 가정했을 때 이상치 포착모형

# In[18]:


# OneClassSVM 
seed = 1234
Eeenv   = EllipticEnvelope(contamination=0.02)
Eeenv.fit(DFX0)


# In[19]:


# 정상치 점수: 점수가 높을수록 inlier, 점수가 낮을수록 outlier
scr_eenv = Eeenv.score_samples(DFX0)
fig, ax = plt.subplots(2,1)
ax[0].hist(scr_eenv);
ax[1].boxplot(scr_eenv, vert=False);


# In[20]:


cutoff = np.quantile(scr_eenv, 0.02)
isin, nx = np.unique(scr_eenv>cutoff, return_counts=True)
cutoff, isin, nx, nx/len(scr_eenv)


# In[21]:


iheenv = Eeenv.predict(DFX0)
from scipy.stats.contingency import crosstab
crosstab(scr_eenv>cutoff, iheenv) 


# In[22]:


DFX0.iloc[iheenv<0]


# ## Isolation Forest
# * `sklearn.ensemble.IsolationForest(n_estimators=100, max_featires=1, contamination=0.1, ...)`
# * 각 샘플에 대하여 Isolation Forest(랜덤으로 고른 변수를 랜덤으로 분리하는 랜덤포레스트)를 적합했을 때 나무들의 깊이의 평균을 이용
# * 이상치는 나무의 초기 분리에 나타나므로 평균깊이가 작을수록 이상치일 가능성이 높음

# In[23]:


# Isolation Forest
seed = 1234
Eisof   = IsolationForest(contamination=0.02, random_state=seed) # max_samples='auto', ... 
Eisof.fit(DFX0)


# In[24]:


# 정상치 점수: 점수가 높을수록 inlier, 점수가 낮을수록 outlier
scr_isof = Eisof.score_samples(DFX0)
fig, ax = plt.subplots(2,1)
ax[0].hist(scr_isof);
ax[1].boxplot(scr_isof, vert=False);


# In[25]:


cutoff = np.quantile(scr_isof, 0.02)
isin, nx = np.unique(scr_isof>cutoff, return_counts=True)
cutoff, isin, nx, nx/len(scr_isof)


# In[26]:


ihisof = Eisof.predict(DFX0)
# from scipy.stats.contingency import crosstab
crosstab(scr_isof>cutoff, ihisof) 


# In[27]:


DFX0.iloc[ihisof<0]


# ## Local Outlier Factor
# * `sklearn.neighbors.LocalOutlierFactor(n_neighbors=20, contamination, novelty=False ..)`: 
# * kNN기준으로 샘플의 확률밀도함수와 그 샘플의 이웃들의 확률밀도함수를 비교. 
# * 이상치는 본인의 밀도가 이웃의 밀도보다 낮음 
# * LOF가 1보다 크면 이상치일 가능성 높음
# * `novelty=False`면 `score_samples` 사용불가. 예측하려면 `fit_predict()`사용 
# * `novelty=True`면 `predict, score_smaples` 사용가능하지만 특이치 포착에 사용할 것. TR에 사용금지

# In[28]:


# Isolation Forest
Elof = LocalOutlierFactor(contamination=0.02) # , novelty=True) # n_neighbors=20
Elof.fit(DFX0)


# In[29]:


ihlof = Elof.fit_predict(DFX0)
isin, nx = np.unique(ihlof, return_counts=True)
isin, nx, nx/len(ihlof)


# ## 이상치 모형결과 비교

# In[30]:


DF0in = DF0[['y']].copy()
DF0in.head() 


# In[31]:


# 예측값: {1:inlier, -1:outlier} 
DF0in['ihonesvm'] = Eonesvm.predict(DFX0)
DF0in['iheenv']   = Eeenv.predict(DFX0)
DF0in['ihisof']   = Eisof.predict(DFX0)
DF0in['ihlof']    = Elof.fit_predict(DFX0)
DF0in['iscore']   = DF0in['ihonesvm'] + DF0in['iheenv'] + DF0in['ihisof'] + DF0in['ihlof']
DF0in['isoutlier']= DF0in['iscore']<=-4  # 4개 모두 이상치로 판단하면 이상치로 간주
DF0in.head()


# In[32]:


DF0in.drop(['y', 'iscore', 'isoutlier'], axis=1).apply(pd.Series.value_counts) 


# In[33]:


DF0in['iscore'].value_counts(), DF0in['iscore'].value_counts(normalize=True)


# In[34]:


DF0.loc[DF0in['isoutlier']==True].sort_values(by=['age','capgain','caploss','hr'], ascending=False)   # 5개 이상치


# In[35]:


DF0.loc[DF0in['iscore']<=-2,:]  # 3개 이상의 모형이 이상치로 판단된 223개 샘플


# In[36]:


sns.heatmap(DF0in.iloc[:,1:5])


# In[37]:


# sns.heatmap(DF0in.iloc[:,1:5].sort_values(by=['ihonesvm','iheenv','ihisof','ihlof']))
sns.heatmap(DF0in.iloc[:,1:5].sort_values(by=['ihonesvm','iheenv','ihisof','ihlof']))


# In[38]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PTpca = make_pipeline(StandardScaler(), PCA(n_components=3))
DFX0pca = PTpca.fit_transform(DFX0)
DFX0pca = pd.DataFrame(DFX0pca, columns=['PC1', 'PC2', 'PC3'])
DFX0pca.head()


# In[39]:


# 주성분별 분산설명력, 누적 분산설명력
PTpca['pca'].explained_variance_ratio_, np.cumsum(PTpca['pca'].explained_variance_ratio_)


# In[40]:


DFX0.head()


# In[41]:


# 주성분계수 (주의: row기준임)
np.round(PTpca['pca'].components_, 3) # age, capgain, caploss, hr 


# In[42]:


fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 12))
sns.scatterplot(x='PC1', y='PC2', hue=DF0in['isoutlier'], data=DFX0pca, ax=axs[0,0], alpha=0.5);
sns.scatterplot(x='PC1', y='PC2', hue=DF0in['iscore']<=-2, data=DFX0pca, ax=axs[0,1], alpha=0.5);
sns.scatterplot(x='PC1', y='PC2', hue=DF0in['ihonesvm']==-1, data=DFX0pca, ax=axs[0,2], alpha=0.5);
sns.scatterplot(x='PC1', y='PC2', hue=DF0in['iheenv']==-1, data=DFX0pca, ax=axs[1,0], alpha=0.5);
sns.scatterplot(x='PC1', y='PC2', hue=DF0in['ihisof']==-1, data=DFX0pca, ax=axs[1,1], alpha=0.5);
sns.scatterplot(x='PC1', y='PC2', hue=DF0in['ihlof']==-1, data=DFX0pca, ax=axs[1,2], alpha=0.5);
plt.show()


# In[43]:


# 3D scatter
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize = (16, 9))
ax  = plt.axes(projection ='3d')
p   = ax.scatter3D(DFX0pca['PC1'], DFX0pca['PC2'], DFX0pca['PC3'], alpha=0.25, c=DF0in['iscore']<=-2) 
ax.set_xlabel('X:PC1')
ax.set_ylabel('Y:PC2')
ax.set_zlabel('Z:PC3');


# In[44]:


# !pip install plotly


# In[45]:


import plotly.express as px 
# 처음 3개의 주성분 점수에 대한 3D 산점도
fig = px.scatter_3d(DFX0pca, x='PC1', y='PC2', z='PC3', color=DF0in['isoutlier'], opacity=0.5)
fig.update_traces(marker_size = 4)
fig.show()


# In[46]:


import plotly.express as px 
# 처음 3개의 주성분 점수에 대한 3D 산점도
fig = px.scatter_3d(DFX0pca, x='PC1', y='PC2', z='PC3', color=DF0in['iscore']<=-2, opacity=0.5)
fig.update_traces(marker_size = 4)
fig.show()


# In[47]:


import plotly.express as px 
# 처음 3개의 주성분 점수에 대한 3D 산점도
fig = px.scatter_3d(DFX0pca, x='PC1', y='PC2', z='PC3', color=DF0in['ihonesvm']==-1, opacity=0.5)
fig.update_traces(marker_size = 4)
fig.show()


# In[48]:


import plotly.express as px 
# 처음 3개의 주성분 점수에 대한 3D 산점도
fig = px.scatter_3d(DFX0pca, x='PC1', y='PC2', z='PC3', color=DF0in['iheenv']==-1, opacity=0.5)
fig.update_traces(marker_size = 4)
fig.show()


# In[49]:


import plotly.express as px 
# 처음 3개의 주성분 점수에 대한 3D 산점도
fig = px.scatter_3d(DFX0pca, x='PC1', y='PC2', z='PC3', color=DF0in['ihisof']==-1, opacity=0.5)
fig.update_traces(marker_size = 4)
fig.show()


# In[50]:


DF0in['ihisof'].value_counts(), DF0in['ihisof'].value_counts(normalize=True)


# In[51]:


DF0in['ihisof'].value_counts().plot(kind='bar');


# In[52]:


import plotly.express as px 
# 처음 3개의 주성분 점수에 대한 3D 산점도
fig = px.scatter_3d(DFX0pca, x='PC1', y='PC2', z='PC3', color=DF0in['ihlof']==-1, opacity=0.5)
fig.update_traces(marker_size = 4)
fig.show()

