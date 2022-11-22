#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
start_time = time.time()


# # Imbalanced data analysis 
# * 목표: SMOTE 예제 
# * [SMOTE with imbalance Data: CreditCard](https://www.kaggle.com/code/qianchao/smote-with-imbalance-data/notebook)
# * [Dealing with Class Imbalance with SMOTE, tf-keras](https://www.kaggle.com/code/theoviel/dealing-with-class-imbalance-with-smote/notebook)
# 
# ## 용어
# 
# * 양성(+1), 성공, 1, 사건(event), 소수클래스(minority class), 드문 사건(rare event)
# * 음성(-1), 실패, 0, 비사건(non-event), 다수클래스(majority class), 흔한 사건(non-rare event)
# * 불균형 자료 imbalanced data: 분류분석에서 타겟의 수준별 빈도(클래스 빈도)가 고르지 않은 자료
# 
# 
# ## 불균형 자료의 문제점
# * 소수클래스 포착이 어려움
# * 일반적으로 분류모형은 정확도(accuracy)를 높이도록 개발됨. 불균형이 심하면 다수클래스쪽으로 편향발생, 소수클래스는 무시하게 됨
# * 다수클래스 예측은 좋아지고(; spec이 높아지고), 소수클래스 예측은 나빠짐(; sens는 낮아짐)
# * 불균형이면 모형 정확도는 매우 높지만, 소수클래스를 다수클래스로 예측할 위험이 커짐(False Negative Rate = 1-Sens = Miss Rate 높아짐)
# * 불균형 자료에서 정확도는 성능지표로 부적절
# * 오분류 비용이 대칭적이면 소수클래스는 무시하고, 다수클래스쪽으로 편향발생 
# * NIR(No Information Rate): 
#      * 다수클래스의 상대빈도. 다수결 모형. 모든 클래스를 다수클래스로 예측했을 때의 적중률. 
#      * 불균형이 심하면 매우 높은 정확도를 보임. 
# * 소수클래스 편향모형 필요: 정확도가 낮아지는 것을 감수하더라도 소수클래스 포착이 가능한 모형 필요
# 
# ## 불균형 자료 접근방식
# 
# * 접근방식
#     * 모형수준(algorithm-level, internal 방식): 소수클래스를 포착할 수 있도록 모형을 개선. class weight, implicit priors, 
#     * 데이터수준(data-level, external 방식): 균형 자료가 되도록 재표본. 모형은 수정없이 사용. Oversmapling, Undersampling, SMOTE-like
#     * 비용민감학습(Cost-sensitive learning): 소수클래스의 오분류비용을 높임. 오분류비용을 감안할 수 있도록 모형을 개선. Cost-sensitive SVM, Cost-sensitive Tree
#     * 앙상블기반 방법(Ensemble-based): 데이터수준 또는 비용민감학습 + 앙상블 모형. RUSBoost, BalancedRandomForest
# 
# 
# ## 불균형자료 형태
# * Small sample size 소수클래스 샘플수가 극히 낮은 경우: 범죄. 이상치포착과 유사. 불균형비율이 높더라도 소수클래스 샘플수가 충분하면 포착가능
# * Overlapping, class separability 클래스간 분리 자체가 어려운 경우: 소수, 다수클래스가 겹쳐서 분리면 찾기 힘든 경우. 
# * Small disjuncts 소수클래스가 군집으로 전역에 퍼져 있는 경우: 
# 

# # 모형기반
# 
# * 가중치나 비용반영 여부
# 
# scikit-learn/xgboost/tensorflow/costcla|class_weight|비용행렬지정여부
# :--------------------------------------|:-----------|:---------------
# LogisticRegression,                    |O|X
# costcla.CostSensitiveLogisticRegression|X|O
# LogisticRegression(penalty='elasticnet')|O|X
# SVC|O|X
# DecisionTreeClassifier|O|X
# CostSenstiveDecisionTreeClassifier|X|O
# Random Forest|O|X
# CostSensitiveRandomClassifier|X|O
# 
# 
# 
# # 데이터수준: 재표본
# * 적합용 자료(TR)를 균형자료가 되도록 재표본한 후 일반적인 분류모형을 적용
# * Oversampling(과대추출법): 소수클래스 과대추출. 소수클래스 샘플을 과표본하여 균형화. 정보손실은 없지만 소수 클래스가 과대대표됨. 계산부담증가
#     * ROS: 소수클래스를 균형이 될 때까지 재표본. 원 샘플이 계속 재표본되므로 중복 불가피
#     * SMOTE: 소수클래스 샘플을 합성하여 균형화. 중복회피 
# * Undersampling(과소추출법): 다수클래스 과소추출. 다수클래스를 랜덤 추출하거나 오분류 위험이 높은 다수 클래스 샘플을 제거하여 균형화
#     * RUS: 다수클래스를 균형이 될 때까지 재표본. 계산부담감소
#     * Tomek link: 클래스가 다른 두 자료 사이에 다른 자료가 존재하지 않으면 두 자료는 Tomek Link. Tomek Link 중 다수클래스 샘플을 제거. 분류면이 명확해짐
# * Oversampling + Undersampling: SMOTE는 합성과정에서 이상치 생성 위험 있음. 소수클래스 합성후 과소추출법으로 다수클래스 축소하여 분류면을 명확히 함    

# # 예제 1: 
#    
# <!--
# ## 자료설명
# 
# * https://matplotlib.org/stable/tutorials/colors/colormaps.html
# 
# 
# ### plotnine: * plotnine: matplotlib기반 ggplot2
# -->
# 
# 
# ```
# ! python -m pip install -U pip
# ! pip install plotnine
# ! pip install imblearn   # Permission오류시 아나콘다 프롬프트를 관리자권한으로 열고 conda activate tf28 후 pip할 것
# # 주의: import sklearn; sklearn.__version__ sklearn이 업그레이드 됨. 기존 작성한 프로그램에서 오류날 수 있음
# ```
# 
# 
# * [sk 자료생성함수예](https://scikit-learn.org/stable/auto_examples/datasets/plot_random_dataset.html)
#     * `make_classification`: 정규분포 분류분석자료 생성. 수준별 혼합분포가능
#     * `make_blobs`: 정규분포 분류분석자료 생성(방울모양, isotropic Gaussian blobs)
#     * `make_gaussian_quantiles`: 정규분포 분류분석자료 생성 
#     * `make_[circles,moons,hastie_10_2]`
#     * `make_multilabel_classification`
#     * `make_[biclusters, checkerboard]`
#     * `make_[regression, friedman1,friedman2,friedman3]`:
#     * `make_[s_curve, swiss_roll]`:
#     * `make_[low_rank_matrix,sparse_coded_signal,spd_matrix, sparse_spd_matrix]`:
#     
# * [imblearn 자료생성함수예]    
# 
# ## 자료
# 
# ### 자료생성

# In[2]:


import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_blobs

X, y = make_classification(n_samples=100,     # 샘플수
                           n_features=2,      # 전체 입력수
                           n_informative=2,   # 타겟과 상관있는 입력수
                           n_redundant=0,     # 타겟과 상관없는  입력수
                           n_classes=2,       # 타겟수준수
                           n_clusters_per_class=1, # 타겟수준별 군집수 (기본값=2)
                           weights=(0.95,0.05), # 수준비율: p(0)=0.9, p(1)=0.1
                           class_sep = 2,     # 타겟수준 분리수준. 값이 클수록 구분이 잘 됨
                           random_state=1111)
# we create two clusters of random points

X, y = make_blobs(n_samples = [2000, 200],                # n0=2000, n1=200
                  centers   = [[0.0, 0.0], [3, 3]],
                  cluster_std = [1.5, 1.5],
                  random_state=1111,
                  shuffle=False,
)
y
np.unique(y, return_counts=True)


# ### 자료탐색

# In[3]:


# 시각화 1: np.array + matplotlib
import matplotlib.pyplot as plt

# color는 색이름 지정 
# s:size, c:col는 제3의 변수 사용가능
# marker='o' 는 제3의 변수 사용불가  # Matplotlib does not accepts different markers per plot.
# plt.scatter(X[:,0], X[:,1], c=np.where(y==0, 'blue', 'red'), alpha=0.5)
plt.scatter(X[:,0], X[:,1], c=y, alpha=0.75)


# In[4]:


mrk = np.where(y==0,'o','x')
col = np.where(y==0,'blue','red')
for i in range(X.shape[0]):
    plt.scatter(X[i,0], X[i,1], marker=mrk[i], c=col[i], alpha=0.75) 


# In[5]:


DF = pd.concat([pd.DataFrame(X), pd.Series(y)], axis=1)
DF.columns=['x1','x2','y']
DF.head()


# In[6]:


from sklearn.model_selection import train_test_split
TRX, TSX, TRy, TSy = train_test_split(DF.drop('y', axis=1), DF['y'], train_size=1500, stratify=DF['y'], random_state=1111)


# In[7]:


k, nk = np.unique(TRy, return_counts=True)
nk/len(TRy)


# In[8]:


k, nk = np.unique(TSy, return_counts=True)
nk/len(TSy)


# In[9]:


TRX.plot(kind='scatter', x='x1', y='x2', c=TRy, cmap='Set1', alpha=0.5)  # marker 변경불가


# In[10]:


# 시각화 1: np.array + matplotlib
import matplotlib.pyplot as plt

# color는 색이름 지정 
# s:size, c:col는 제3의 변수 사용가능
# marker='o' 는 제3의 변수 사용불가  # Matplotlib does not accepts different markers per plot.
# plt.scatter(X[:,0], X[:,1], c=np.where(y==0, 'blue', 'red'), alpha=0.5)
plt.scatter(TRX.iloc[:,0], TRX.iloc[:,1], c=TRy, alpha=0.75)


# In[11]:


# 시각화 2: pd + pd.plot  . marker 변경불가 
TRX.plot(kind='scatter', x='x1', y='x2', c=TRy, cmap='Set1', alpha=0.75)  # marker 변경불가


# In[12]:


import seaborn as sns
sns.scatterplot(data=TRX, x='x1', y='x2', hue=TRy, style=TRy)


# In[13]:


# plotnine
from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap
from plotnine.data import mtcars

ggplot(TRX, aes(x='x1', y='x2', color='factor(TRy)', shape='factor(TRy)')) + geom_point()


# ## 이진분류 성능지표

# In[14]:


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score 
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

from imblearn.metrics import geometric_mean_score 
from imblearn.metrics import classification_report_imbalanced 



def metcls2(y, ph, yh, modelname=None):

    tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
    acc = accuracy_score(y, yh)
    balacc = balanced_accuracy_score(y, yh)
    kap   = cohen_kappa_score(y, yh)
    f1    = f1_score(y, yh) # 2 x (prec x rec)/(prec+rec) = harmonic avg(prec, rec)
    gmean = geometric_mean_score(y, yh)

    rec  = recall_score(y, yh) 
    prec = precision_score(y, yh) 
    spec = tn/(tn+fp) 
    iba  = (1+0.1*(rec-spec))*rec*spec  # alpha=0.1일때 index of balanced accuracy

    rocauc  = roc_auc_score(y, ph)
    avgprec = average_precision_score(y, ph)

    metrics = {
        'acc':acc, 'kappa':kap, 'f1':f1, 'gmean':gmean, 'iba':iba, 'balacc':balacc,
        'prec':prec, 'rec':rec, 'spec':spec,
        'rocauc':rocauc, 'avgprec':avgprec,
        'tn[0,0]':tn,  'fp[0,1]':fp,    
        'fn[1,0]':fn,  'tp[1,1]':tp
    }
    if modelname != None:
        return pd.DataFrame(metrics, index=[modelname])
    return metrics   


# ## Base model

# In[15]:


from sklearn.linear_model import LogisticRegression
Eglm = LogisticRegression()
Eglm.fit(TRX, TRy)
Eglm.score(TRX, TRy), Eglm.coef_, Eglm.intercept_


# In[16]:


COEF = pd.DataFrame((np.c_[Eglm.intercept_, 
                           Eglm.coef_, 
                           -Eglm.intercept_/Eglm.coef_[0,1],
                           -Eglm.coef_[0,0]/Eglm.coef_[0,1]]), columns=['b0', 'b1', 'b2', 'const', 'slope'], index=['glm'] )
# COEF.slope = b1glm = - Eglm.coef_[0,0]/Eglm.coef_[0,1]
COEF


# In[17]:


print(classification_report_imbalanced (TRy, Eglm.predict(TRX)))


# In[18]:


metcls2(TRy, Eglm.predict_proba(TRX)[:,1], Eglm.predict(TRX)), metcls2(TSy, Eglm.predict_proba(TSX)[:,1], Eglm.predict(TSX))


# In[19]:


TRMET = pd.DataFrame(metcls2(TRy, Eglm.predict_proba(TRX)[:,1], Eglm.predict(TRX)), index=['glm'])
TSMET = pd.DataFrame(metcls2(TSy, Eglm.predict_proba(TSX)[:,1], Eglm.predict(TSX)), index=['glm'])
TSMET


# In[20]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4)) 
ConfusionMatrixDisplay(confusion_matrix(TRy, Eglm.predict(TRX))).plot(ax=ax1)
ConfusionMatrixDisplay(confusion_matrix(TSy, Eglm.predict(TSX))).plot(ax=ax2)


# In[21]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4)) 
fpr, tpr, thres = roc_curve(TRy, Eglm.predict_proba(TRX)[:,1], pos_label=1)
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax1)
fpr, tpr, thres = roc_curve(TSy, Eglm.predict_proba(TSX)[:,1], pos_label=1)
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax2)


# In[22]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4)) 
prec, rec, thres = precision_recall_curve(TRy, Eglm.predict_proba(TRX)[:,1], pos_label=1)
PrecisionRecallDisplay(precision=prec, recall=rec).plot(ax=ax1)
prec, rec, thres = precision_recall_curve(TSy, Eglm.predict_proba(TSX)[:,1], pos_label=1)
PrecisionRecallDisplay(precision=prec, recall=rec).plot(ax=ax2)


# In[23]:


COEF.loc['glm','b0'],  COEF.loc['glm',['b1','b2']]


# In[24]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4)) 
sns.scatterplot(data=TRX, x='x1', y='x2', hue=TRy, style=TRy, ax=ax1)
ax1.axline((0, COEF.loc['glm','const']), slope=COEF.loc['glm','slope']) 

sns.scatterplot(data=TSX, x='x1', y='x2', hue=TSy, style=TSy, ax=ax2)
ax2.axline((0, COEF.loc['glm','const']), slope=COEF.loc['glm','slope']) 
#plt.axline((0, b0glm[0]), slope=b1glm, ax=ax2)


# ##  'class_weight'

# In[25]:


# class_weight = 'balanced'
from sklearn.linear_model import LogisticRegression
Eglmcls = LogisticRegression(class_weight='balanced')
Eglmcls.fit(TRX, TRy)
Eglmcls.score(TRX, TRy), Eglmcls.coef_, Eglmcls.intercept_


# In[26]:


TRMET = pd.concat([TRMET, pd.DataFrame(metcls2(TRy, Eglmcls.predict_proba(TRX)[:,1], Eglmcls.predict(TRX)), index=['glmcls'])])
TSMET = pd.concat([TSMET, pd.DataFrame(metcls2(TSy, Eglmcls.predict_proba(TSX)[:,1], Eglmcls.predict(TSX)), index=['glmcls'])])


# In[27]:


TRMET   # acc, kappa, f1 감소, gmean, iba, balacc 증가, prec, spec 감소, rec 증가, rocauc, avgprec 비슷 => tp급증, fp급증


# In[28]:


TSMET


# In[29]:


print(classification_report_imbalanced(TSy, Eglmcls.predict(TSX)))


# In[30]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4)) 
ConfusionMatrixDisplay(confusion_matrix(TRy, Eglmcls.predict(TRX))).plot(ax=ax1)
ConfusionMatrixDisplay(confusion_matrix(TSy, Eglmcls.predict(TSX))).plot(ax=ax2)


# In[31]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4)) 
fpr, tpr, thres = roc_curve(TRy, Eglmcls.predict_proba(TRX)[:,1], pos_label=1)
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax1)
fpr, tpr, thres = roc_curve(TSy, Eglmcls.predict_proba(TSX)[:,1], pos_label=1)
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax2)


# In[32]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4)) 
prec, rec, thres = precision_recall_curve(TRy, Eglmcls.predict_proba(TRX)[:,1], pos_label=1)
PrecisionRecallDisplay(precision=prec, recall=rec).plot(ax=ax1)
prec, rec, thres = precision_recall_curve(TSy, Eglmcls.predict_proba(TSX)[:,1], pos_label=1)
PrecisionRecallDisplay(precision=prec, recall=rec).plot(ax=ax2)


# In[33]:


COEF = pd.concat([COEF, 
                  pd.DataFrame(
                      np.c_[Eglmcls.intercept_, 
                            Eglmcls.coef_,  
                            -Eglmcls.intercept_/Eglmcls.coef_[0,1], 
                            -Eglmcls.coef_[0,0]/Eglmcls.coef_[0,1]], 
                      columns=['b0', 'b1', 'b2', 'const', 'slope'], index=['glmcls'] )], axis=0)
COEF


# In[34]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4)) 
sns.scatterplot(data=TRX, x='x1', y='x2', hue=TRy, style=TRy, ax=ax1)
ax1.axline((0, COEF.loc['glm','const']), slope=COEF.loc['glm','slope'], linestyle='--') 
ax1.axline((0, COEF.loc['glmcls','const']), slope=COEF.loc['glmcls','slope']) 

sns.scatterplot(data=TSX, x='x1', y='x2', hue=TSy, style=TSy, ax=ax2)
ax2.axline((0, COEF.loc['glm','const']), slope=COEF.loc['glm','slope'], linestyle='--') 
ax2.axline((0, COEF.loc['glmcls','const']), slope=COEF.loc['glmcls','slope']) 
#plt.axline((0, b0glm[0]), slope=b1glm, ax=ax2)


# ## Sampling 을 이용한 불균형자료분석
# ### ROS: RandomOverSampler
# 

# In[35]:


# TRO: TR Oversampled
from imblearn.over_sampling import RandomOverSampler
Tros = RandomOverSampler(random_state=1111)
TROX, TROy = Tros.fit_resample(TRX, TRy)
TROX.shape, TROy.value_counts()


# In[36]:


# sns.scatterplot(data=DFros, x='x1', y='x2', hue='y', style='y') # sns는 절편, 기울기방식의 선을 그릴수 없음
# ROS는 기존 샘플을 계속 재표집하므로, 그림상으로 확인안됨
TROX.plot(kind='scatter', x='x1', y='x2', c=TROy, cmap='Set1')  


# In[37]:


# glm on ROS
Eglmros = LogisticRegression()
Eglmros.fit(TROX, TROy)
Eglmros.score(TROX, TROy), Eglmros.coef_, Eglmros.intercept_


# In[38]:


TRMET = pd.concat([TRMET, pd.DataFrame(metcls2(TRy, Eglmros.predict_proba(TRX)[:,1], Eglmros.predict(TRX)), index=['glmros'])])
TSMET = pd.concat([TSMET, pd.DataFrame(metcls2(TSy, Eglmros.predict_proba(TSX)[:,1], Eglmros.predict(TSX)), index=['glmros'])])
TRMET


# In[39]:


TSMET


# In[40]:


print(classification_report_imbalanced(TRy, Eglmros.predict(TRX)))


# In[41]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4)) 
ConfusionMatrixDisplay(confusion_matrix(TRy, Eglmros.predict(TRX))).plot(ax=ax1)
ConfusionMatrixDisplay(confusion_matrix(TSy, Eglmros.predict(TSX))).plot(ax=ax2)


# In[42]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4)) 
fpr, tpr, thres = roc_curve(TRy, Eglmros.predict_proba(TRX)[:,1], pos_label=1)
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax1)
fpr, tpr, thres = roc_curve(TSy, Eglmros.predict_proba(TSX)[:,1], pos_label=1)
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax2)


# In[43]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4)) 
prec, rec, thres = precision_recall_curve(TRy, Eglmros.predict_proba(TRX)[:,1], pos_label=1)
PrecisionRecallDisplay(precision=prec, recall=rec).plot(ax=ax1)
prec, rec, thres = precision_recall_curve(TSy, Eglmros.predict_proba(TSX)[:,1], pos_label=1)
PrecisionRecallDisplay(precision=prec, recall=rec).plot(ax=ax2)


# In[44]:


COEF = pd.concat([COEF, 
                  pd.DataFrame(
                      np.c_[ Eglmros.intercept_, 
                             Eglmros.coef_,  
                            -Eglmros.intercept_/Eglmros.coef_[0,1], 
                            -Eglmros.coef_[0,0]/Eglmros.coef_[0,1]], 
                      columns=['b0', 'b1', 'b2', 'const', 'slope'], index=['glmros'] )], axis=0)
COEF


# In[45]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4)) 
sns.scatterplot(data=TRX, x='x1', y='x2', hue=TRy, style=TRy, ax=ax1)
ax1.axline((0, COEF.loc['glm','const']), slope=COEF.loc['glm','slope'], linestyle='--') 
ax1.axline((0, COEF.loc['glmcls','const']), slope=COEF.loc['glmcls','slope'], linestyle=':') 
ax1.axline((0, COEF.loc['glmros','const']), slope=COEF.loc['glmros','slope']) 

sns.scatterplot(data=TSX, x='x1', y='x2', hue=TSy, style=TSy, ax=ax2)
ax2.axline((0, COEF.loc['glm','const']), slope=COEF.loc['glm','slope'], linestyle='--') 
ax2.axline((0, COEF.loc['glmcls','const']), slope=COEF.loc['glmcls','slope'], linestyle=':') 
ax2.axline((0, COEF.loc['glmros','const']), slope=COEF.loc['glmros','slope']) 
#plt.axline((0, b0glm[0]), slope=b1glm, ax=ax2)


# ### SMOTE: 합성

# In[46]:


# SMOTE
from imblearn.over_sampling import SMOTE, ADASYN
Esmt = SMOTE(random_state=1111)
TRSX, TRSy = Esmt.fit_resample(TRX, TRy)  # 동일한 샘플이 중복되어 추출되므로, 그림상으로 확인불가
TRSX.shape, TRSy.value_counts()


# In[47]:


# sns.scatterplot(data=DFros, x='x1', y='x2', hue='y', style='y') # sns는 절편, 기울기방식의 선을 그릴수 없음
# SMOTE는 기존 샘플을 합성하여 추가하므로 그림상으로 확인가능
# TRSX.plot(kind='scatter', x='x1', y='x2', c=TRSy, cmap='Set1')  
sns.scatterplot(data=TRSX, x='x1', y='x2', hue=TRSy, style=TRSy)


# In[48]:


# glm on SMOTE
Eglmsmt = LogisticRegression()
Eglmsmt.fit(TRSX, TRSy)
Eglmsmt.score(TRSX, TRSy), Eglmsmt.coef_, Eglmsmt.intercept_


# In[49]:


TRMET = pd.concat([TRMET, pd.DataFrame(metcls2(TRy, Eglmsmt.predict_proba(TRX)[:,1], Eglmsmt.predict(TRX)), index=['glmsmt'])])
TSMET = pd.concat([TSMET, pd.DataFrame(metcls2(TSy, Eglmsmt.predict_proba(TSX)[:,1], Eglmsmt.predict(TSX)), index=['glmsmt'])])
TRMET


# In[50]:


TSMET


# In[51]:


print(classification_report_imbalanced(TSy, Eglmsmt.predict(TSX)))


# In[52]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4)) 
ConfusionMatrixDisplay(confusion_matrix(TRy, Eglmsmt.predict(TRX))).plot(ax=ax1)
ConfusionMatrixDisplay(confusion_matrix(TSy, Eglmsmt.predict(TSX))).plot(ax=ax2)


# In[53]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4)) 
fpr, tpr, thres = roc_curve(TRy, Eglmsmt.predict_proba(TRX)[:,1], pos_label=1)
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax1)
fpr, tpr, thres = roc_curve(TSy, Eglmsmt.predict_proba(TSX)[:,1], pos_label=1)
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax2)


# In[54]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4)) 
prec, rec, thres = precision_recall_curve(TRy, Eglmsmt.predict_proba(TRX)[:,1], pos_label=1)
PrecisionRecallDisplay(precision=prec, recall=rec).plot(ax=ax1)
prec, rec, thres = precision_recall_curve(TSy, Eglmsmt.predict_proba(TSX)[:,1], pos_label=1)
PrecisionRecallDisplay(precision=prec, recall=rec).plot(ax=ax2)


# In[55]:


COEF = pd.concat([COEF, 
                  pd.DataFrame(
                      np.c_[ Eglmsmt.intercept_, 
                             Eglmsmt.coef_,  
                            -Eglmsmt.intercept_/Eglmsmt.coef_[0,1], 
                            -Eglmsmt.coef_[0,0]/Eglmsmt.coef_[0,1]], 
                      columns=['b0', 'b1', 'b2', 'const', 'slope'], index=['glmsmt'] )], axis=0)
COEF


# In[56]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4)) 
sns.scatterplot(data=TRSX, x='x1', y='x2', hue=TRSy, style=TRy, ax=ax1)
ax1.axline((0, COEF.loc['glm','const']), slope=COEF.loc['glm','slope'], linestyle='--') 
ax1.axline((0, COEF.loc['glmcls','const']), slope=COEF.loc['glmcls','slope'], linestyle=':') 
ax1.axline((0, COEF.loc['glmros','const']), slope=COEF.loc['glmros','slope'], linestyle='-.') 
ax1.axline((0, COEF.loc['glmsmt','const']), slope=COEF.loc['glmsmt','slope']) 

sns.scatterplot(data=TSX, x='x1', y='x2', hue=TSy, style=TSy, ax=ax2)
ax2.axline((0, COEF.loc['glm','const']), slope=COEF.loc['glm','slope'], linestyle='--') 
ax2.axline((0, COEF.loc['glmcls','const']), slope=COEF.loc['glmcls','slope'], linestyle=':') 
ax2.axline((0, COEF.loc['glmros','const']), slope=COEF.loc['glmros','slope'], linestyle='-.') 
ax2.axline((0, COEF.loc['glmsmt','const']), slope=COEF.loc['glmsmt','slope']) 
#plt.axline((0, b0glm[0]), slope=b1glm, ax=ax2)


# ### RUS: RandomUnderSampler

# In[57]:


# RUS
from imblearn.under_sampling import RandomUnderSampler 
Erus = RandomUnderSampler(random_state=1111)
TRUX, TRUy = Erus.fit_resample(TRX, TRy)  # 동일한 샘플이 중복되어 추출되므로, 그림상으로 확인불가
TRUX.shape, TRUy.value_counts()


# In[58]:


sns.scatterplot(data=TRUX, x='x1', y='x2', hue=TRUy, style=TRUy)


# In[59]:


# glm on SMOTE
Eglmrus = LogisticRegression()
Eglmrus.fit(TRUX, TRUy)
Eglmrus.score(TRUX, TRUy), Eglmrus.coef_, Eglmrus.intercept_


# In[60]:


TRMET = pd.concat([TRMET, pd.DataFrame(metcls2(TRy, Eglmrus.predict_proba(TRX)[:,1], Eglmsmt.predict(TRX)), index=['glmrus'])])
TSMET = pd.concat([TSMET, pd.DataFrame(metcls2(TSy, Eglmrus.predict_proba(TSX)[:,1], Eglmrus.predict(TSX)),  index=['glmrus'])])
TRMET


# In[61]:


TSMET


# In[62]:


print(classification_report_imbalanced(TSy, Eglmrus.predict(TSX)))


# In[63]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4)) 
ConfusionMatrixDisplay(confusion_matrix(TRy, Eglmrus.predict(TRX))).plot(ax=ax1)
ConfusionMatrixDisplay(confusion_matrix(TSy, Eglmrus.predict(TSX))).plot(ax=ax2)


# In[64]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4)) 
fpr, tpr, thres = roc_curve(TRy, Eglmrus.predict_proba(TRX)[:,1], pos_label=1)
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax1)
fpr, tpr, thres = roc_curve(TSy, Eglmrus.predict_proba(TSX)[:,1], pos_label=1)
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax2)


# In[65]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4)) 
prec, rec, thres = precision_recall_curve(TRy, Eglmrus.predict_proba(TRX)[:,1], pos_label=1)
PrecisionRecallDisplay(precision=prec, recall=rec).plot(ax=ax1)
prec, rec, thres = precision_recall_curve(TSy, Eglmrus.predict_proba(TSX)[:,1], pos_label=1)
PrecisionRecallDisplay(precision=prec, recall=rec).plot(ax=ax2)


# In[66]:


COEF = pd.concat([COEF, 
                  pd.DataFrame(
                      np.c_[ Eglmrus.intercept_, 
                             Eglmrus.coef_,  
                            -Eglmrus.intercept_/Eglmrus.coef_[0,1], 
                            -Eglmrus.coef_[0,0]/Eglmrus.coef_[0,1]], 
                      columns=['b0', 'b1', 'b2', 'const', 'slope'], index=['glmrus'] )], axis=0)
COEF


# In[67]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4)) 
sns.scatterplot(data=TRSX, x='x1', y='x2', hue=TRSy, style=TRy, ax=ax1)
ax1.axline((0, COEF.loc['glm','const']), slope=COEF.loc['glm','slope'], linestyle='--') 
ax1.axline((0, COEF.loc['glmcls','const']), slope=COEF.loc['glmcls','slope'], linestyle=':') 
ax1.axline((0, COEF.loc['glmros','const']), slope=COEF.loc['glmros','slope'], linestyle='-.') 
ax1.axline((0, COEF.loc['glmrus','const']), slope=COEF.loc['glmrus','slope']) 

sns.scatterplot(data=TSX, x='x1', y='x2', hue=TSy, style=TSy, ax=ax2)
ax2.axline((0, COEF.loc['glm','const']), slope=COEF.loc['glm','slope'], linestyle='--') 
ax2.axline((0, COEF.loc['glmcls','const']), slope=COEF.loc['glmcls','slope'], linestyle=':') 
ax2.axline((0, COEF.loc['glmros','const']), slope=COEF.loc['glmros','slope'], linestyle='-.') 
ax2.axline((0, COEF.loc['glmrus','const']), slope=COEF.loc['glmrus','slope']) 
#plt.axline((0, b0glm[0]), slope=b1glm, ax=ax2)


# ## 모형비교

# In[68]:


# 주요지표를  TR, TS에서 비교 
# 주요지표 acc, f1, gmean, iba, balacc, prec, rec
tmp1 = TRMET.copy()
tmp1['trts'] = 'TR'
tmp2 = TSMET.copy()
tmp2['trts'] = 'TS'
tmp = pd.concat([tmp1, tmp2], axis=0)
tmp


# In[69]:


tmp.groupby('trts').plot(y=['rec', 'prec'], use_index=True, kind='bar')

