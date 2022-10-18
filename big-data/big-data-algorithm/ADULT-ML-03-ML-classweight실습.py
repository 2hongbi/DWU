#!/usr/bin/env python
# coding: utf-8

# # ADULT-ML-03-ML Models 실습
# * 데이터 처리
#     * adult.csv: 원자료
#     * adult4ml.csv : 원자료의 범주형변수 재그룹화한 자료
#     * adult4ml-clean.csv: adult4ml에서 이상치 591개 제거한 자료. fnlwgt, eduyr 제거

# 

# # 자료읽기
# 
# ## 다운로드/CSV 저장
# * `sklearn.datasets.fetch_openml`: 
#     * UCI에서 직접 다운로드. Bunch 또는 DF로 다운 가능
#     * Bunch{data(X), target(y), DESCR} 

# In[1]:


import pandas as pd
import numpy as np
np.set_printoptions(edgeitems=30, precision=4, linewidth=120, sign =' ')
# from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [5, 4] # [6.4, 4.8] default size 
import seaborn as sns


# In[2]:


adultml = pd.read_csv('D:/WS2022B/dataset/adult4ml.csv')  
adultml.head()
adultml.shape


# ## 자료정리
# ###  DF 정리

# In[3]:


# 원본 파일 
DF = adultml.copy()
DF.info()  # numeric, object(문자)로 구성됨 


# In[4]:


# fnlwgt, eduyr 제거  
DF.drop(['fnlwgt', 'eduyr'], axis=1, inplace=True)
vobj = list(DF.select_dtypes('object'))
vnum = list(DF.select_dtypes(np.number))
vord = ['edugrp']
vobj, vnum, vord


# In[5]:


# 문자변수 일괄 category화.
for vname in vobj:
    DF[vname] = DF[vname].astype('category') 

DF['edugrp'] = pd.Categorical(DF['edugrp'], ordered=True)
DF.info()


# In[6]:


DF['y'] = DF['y'].map({'<=50K': 0, '>50K': 1})
# 개별모형에는 지장없으나 cross_val_score에서 score계산시 오류발생
DF.head()


# In[7]:


DF['y'].value_counts()


# # 기본모형
# ## 

# In[8]:


#  결측확인
DF.isnull().sum()


# In[9]:


# from pandas import set_option
# from pandas.tools.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import log_loss

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier


# In[10]:


TRX, TSX, TRy, TSy = train_test_split(DF.drop('y', axis=1), DF['y'],
                                      test_size=0.25, random_state=1234, stratify=DF['y'])
TRX.shape, TSX.shape


# In[11]:


# StratifiedKFold : 층화 CV Splitter. 분류분석시 타겟수준에 대한 층화CV분할 
# n_split : 폴드수 
# shuffle : False(기본값) 면 random_state 지정못함. 
# random_state : 랜덤시드

SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# In[12]:


from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer

# from sklearn import set_config
# set_config(display='diagram')

# 개별변수 변환용 파이프라인 (예시)
# Plog = Pipeline([('impmed', SimpleImputer(strategy='median')),
#                  ('log', FunctionTransformer(np.log)), 
#                 ('normalize', StandardScaler()) ])

# 수치형 변수 공통
Pnum = Pipeline([('impmed', SimpleImputer(strategy='median')), 
                 ('normalize', StandardScaler()) ])

# 범주형 변수 공통
Pcat = Pipeline([('impmod', SimpleImputer(strategy='most_frequent')),
                 ('dummy', OneHotEncoder(handle_unknown='ignore', sparse=False)) ]) # LDA, NB는 sparse 안됨. drop='First'면 R

PP = ColumnTransformer([('num', Pnum, make_column_selector(dtype_include=np.number)),
                        ('cat', Pcat, make_column_selector(dtype_include='category'))], # 또는 지정. ['ocn']
                        remainder=Pnum)    
# from sklearn.compose import make_column_selector, make_column_transformer
# preprocessing = make_column_transformer(
#    (Pnum, make_column_selector(dtype_include=np.number)),
#    (Pcat, make_column_selector(dtype_include=object)),
#)
PP.fit(TRX)


# In[13]:


# Monkey patch
def monkey_patch_get_signature_names_out():
    """Monkey patch some classes which did not handle get_feature_names_out()
       correctly in 1.0.0."""
    from inspect import Signature, signature, Parameter
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    default_get_feature_names_out = StandardScaler.get_feature_names_out

    if not hasattr(SimpleImputer, "get_feature_names_out"):
      print("Monkey-patching SimpleImputer.get_feature_names_out()")
      SimpleImputer.get_feature_names_out = default_get_feature_names_out

    if not hasattr(FunctionTransformer, "get_feature_names_out"):
        print("Monkey-patching FunctionTransformer.get_feature_names_out()")
        orig_init = FunctionTransformer.__init__
        orig_sig = signature(orig_init)

        def __init__(*args, feature_names_out=None, **kwargs):
            orig_sig.bind(*args, **kwargs)
            orig_init(*args, **kwargs)
            args[0].feature_names_out = feature_names_out

        __init__.__signature__ = Signature(
            list(signature(orig_init).parameters.values()) + [
                Parameter("feature_names_out", Parameter.KEYWORD_ONLY)])

        def get_feature_names_out(self, names=None):
            if self.feature_names_out is None:
                return default_get_feature_names_out(self, names)
            elif callable(self.feature_names_out):
                return self.feature_names_out(names)
            else:
                return self.feature_names_out

        FunctionTransformer.__init__ = __init__
        FunctionTransformer.get_feature_names_out = get_feature_names_out

monkey_patch_get_signature_names_out()


# In[14]:


PP.get_feature_names_out() 


# # 모형적합 glm, lasso, ridge, elasticnet 만 시도
# ## glm: LogisticRegression
# 
# 
# * 로지스틱회귀모형의 적합: MLE(최대가능도 추정법 = min log_loss 추정법 = min KL divergence)
# 
# $$\text{ log loss } = -l(y, p) = -(y \log (p) + (1 - y) \log (1 - p)) = \text{ NLL: Negative Log Likelihood}$$
# 
# $$ \min_{w} C \sum_{i=1}^n \left(-y_i \log(\hat{p}(X_i)) - (1 - y_i) \log(1 - \hat{p}(X_i))\right) + r(w).$$
# 
# 
# penalty|$$r(w)$$   |모형
# :------ |:--------------|:-------------------------------------------
# none    | $$0$$         | Logit
# $$\ell_1$$|$$\|w\|_1$$  | L1 logit, lasso logit
# $$\ell_2$$|$$\frac{1}{2}||w ||_2^2 = \frac{1}{2}w^T w$$ | L2 logit, ridge logit
# elasticnet|$$\frac{1 - \rho}{2}w^T w + \rho || w ||_1$$ | $\rho$(`l1_ratio`). `l1_ratio=0`이면 Ridge, `l1_ratio=1`이면 Lasso   
# 
# 
# * `sklearn.linear_model.LogisticRegression(penalty='l2', C=1.0, class_weight=None, solver='lbfgs', max_iter=100, l1_ratio=None, ..)`
# 
# args|설명
# :------|:----------------------------
# `fit_intercept=True`|절편
# `penalty='l2'`|`'l1,l2,elasticnet,none'`. 기본값 L2 벌점 로지스틱회귀 (Ridge).  
# `class_weight=None`|`none,balanced`. 관측값 가중치. `balanced`면 가중치 n_samples/(n_classes * np.bincount(y)) 
# `solver='lbfgs'`|`'newton-cg,lbfgs,liblinear,sag,saga'`: 데이터가 적으면 `liblinear`, 많으면 `sag,saga`. `liblinear`는 다진분류시 OvR만 가능.
# &nbsp; |`l1`은  `liblinear, saga`, `elasticnet`은 `saga`만 가능
# `l1ratio=None`|`elasticnet`일때만 사용. L1 벌점비중. `l1ratio=0` 은 `penalty='l2'`, `l1ratio=1`은 `penalty='l1'`에 해당
# `max_iter=100`|최대 반복수
# `multi_class='auto'`|`ovr,multinomial`: OvR(One vs Rest: 타겟의 수준별로 이진분류모형생성), multinomial:다항분포 추정법
# `C=1.0`| 벌점의 역수 (1/Lambda). 값이 작을수록 강한 규제
# 
# 
# 
# attribs|설명
# :------|:---------------------------
# `coef_`|회귀계수 (절편제외)
# `intercept_`|절편
# `n_features_in`|특성개수 
# `feature_names_in`|특성이름
# 
# 
# method|설명
# :-----|:----------------
# `fit(X, y, sample_weight=None)`|적합
# `predict(X)`|예측값 반환
# `predict_proba(X)`|사후확률값 반환
# `score(X, y, sample_weight=None)`|스코어. 적중률 반환
# 
# 
# ### PEglm: 
# * `LogisticRegression(penalty='l2')`: L2 벌점회귀(Ridge 회귀, 벌점계수 C=1)가 기본모형. 
# * `penalty='none'` 지정 

# * 모형적합

# In[15]:


from sklearn.linear_model import LogisticRegression
# penalty='l2'가 기본(Ridge Logit) => penalty='none' 으로 ordinary Logit 적합 (튜닝모수없음)
PEglm = make_pipeline(PP, LogisticRegression(penalty='none', max_iter=1000))   # class_weight='balanced'
PEglm = Pipeline([('PP', PP), 
                  ('glm', LogisticRegression(penalty='none', max_iter=1000))]) # class_weight='balanced'
# fit and score
PEglm.fit(TRX, TRy)
PEglm.score(TRX, TRy), PEglm.score(TSX, TSy)


# * 모형예측값 저장

# In[16]:


TROUT = pd.DataFrame(TRy.copy())
TROUT['yhglm'] = PEglm.predict(TRX)              
TROUT['phglm'] = PEglm.predict_proba(TRX)[:,1]   
TSOUT = pd.DataFrame(TSy.copy())
TSOUT['yhglm'] = PEglm.predict(TSX) 
TSOUT['phglm'] = PEglm.predict_proba(TSX)[:,1] 
TSOUT.head()


# * 모형평가

# In[17]:


# 평가지표 계산함수
# 주의 y는 {0, 1}로 코딩할 것
def metbin(y, yh, ph=None, modelname=None):
    tn, fp, fn, tp = confusion_matrix(y, yh).ravel() 
    acc    = accuracy_score(y, yh)  
    kap    = cohen_kappa_score(y, yh)
    f1     = f1_score(y, yh)
    balacc = balanced_accuracy_score(y, yh) 
    # theshold=0.5일때 tn, fp, fn, tp로 직접 계산 
    sens = tp/(tp+fn)  # recall_score(y, yhsvm) 
    spec = tn/(tn+fp)
    prec = tp/(tp+fp)  # precision_score(y, yh) 
    
    if ph is not None:
        aucpr  = average_precision_score(y, ph)
        aucroc = roc_auc_score(y, ph)
        metrics =  {'acc' :acc, 'sens':sens,    'spec':spec,     'prec':prec, 
                    'f1'  :f1,  'aucpr': aucpr, 'aucroc': aucroc,
                    'tn'  :tn,  'fp'  :fp,      'fn'  :fn,       'tp'  :tp}
    else:
        metrics =  {'acc' :acc, 'sens':sens,    'spec':spec,     'prec':prec, 'f1'  :f1,  
                    'tn'  :tn,  'fp'  :fp,      'fn'  :fn,       'tp'  :tp}
    if modelname is not None:
        return pd.DataFrame(metrics, index=[modelname])
    return metrics


# In[18]:


# confusion_matrix: From-To 방식 {0:실패, 1:성공}
# tn, fp, fn, tp = confusion_matrix(y, yh).ravel() 로 개별원소접근
confusion_matrix(TROUT['y'], TROUT['yhglm'])


# In[19]:


# classification(y, yh) : 
# 수준별 {precision, recall, f1-score, support(데이터수)}
# 전체기준 accuracy (=Micro avg accuracy)
# macro avg: 수준별 지표의 단순평균. macro avg accuracy 는 0의 적중률과 1의 적중률의 단순평균(불균형자료에서 소수클래스가 중요할때 권장)
# weighted avg: 수준별 지표의 가중평균(가중치는 수준별 support). 불균형자료애서 다수클래스가 중요할때 권장 
print(classification_report(TROUT['y'], TROUT['yhglm']) )


# In[20]:


# 다수 지표 한번에 계산
scr = {'acc' :'accuracy',
       'aucpr': 'average_precision',
       'aucroc': 'roc_auc',
       'f1': 'f1',
       'rec': 'recall',
       'prec': 'precision'}
CV = pd.DataFrame(cross_validate(PEglm, TRX, TRy, scoring=scr, cv=SKF))  # dict반환. 5-fold CV결과
# CVMET: 각 모형의 5-fold CV 결과 평균값 저장
CV = CV.filter(regex='test').mean()   # dplyr::filter와 다름 
CVMET = CV.to_frame(name='glm').T
CVMET


# In[21]:


TRMET = metbin(TROUT['y'], TROUT['yhglm'], TROUT['phglm'], modelname='TRglm')
TSMET = metbin(TSOUT['y'], TSOUT['yhglm'], TSOUT['phglm'], modelname='TSglm')
pd.concat([TRMET, TSMET], axis=0)


# * 모형평가 시각화

# In[22]:


fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10,5), constrained_layout=True)
# 기본마진 fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2) 

display = ConfusionMatrixDisplay.from_predictions(TROUT['y'], TROUT['yhglm'], ax=ax[0,0])
display.ax_.set_title('ConfusionMatrixDisplay(TR)');


# Estimator를 이용하려면 RocCurveDisplay.from_estimator(estimator, X, y, ..)
# ph이 있으면 RocCurveDisplay.from_predictions(y, ph, ...)
from sklearn.metrics import RocCurveDisplay
display = RocCurveDisplay.from_predictions(TROUT['y'], TROUT['phglm'], ax=ax[0,1]);
display.ax_.set_title('RocCurveDispaly glm on TR');


# Estimator를 이용하려면 PrecisionRecallDisplay.from_estimator(E, X, y, ..)
# ph이 있으면 PrecisionRecallDisplay.from_predicitions(y, ph, ..)
from sklearn.metrics import PrecisionRecallDisplay
display = PrecisionRecallDisplay.from_predictions(TROUT['y'], TROUT['phglm'], ax=ax[0,2]);
display.ax_.set_title('PrecisionRecallDisplay glm on TR');


display = ConfusionMatrixDisplay.from_predictions(TSOUT['y'], TSOUT['yhglm'], ax=ax[1,0])
display.ax_.set_title('ConfusionMatrixDisplay glm on TS');


# Estimator를 이용하려면 RocCurveDisplay.from_estimator(estimator, X, y, ..)
# ph이 있으면 RocCurveDisplay.from_predictions(y, ph, ...)
from sklearn.metrics import RocCurveDisplay
display = RocCurveDisplay.from_predictions(TSOUT['y'], TSOUT['phglm'], ax=ax[1,1]);
display.ax_.set_title('RocCurveDispaly glm on TS');


# Estimator를 이용하려면 PrecisionRecallDisplay.from_estimator(E, X, y, ..)
# ph이 있으면 PrecisionRecallDisplay.from_predicitions(y, ph, ..)
from sklearn.metrics import PrecisionRecallDisplay
display = PrecisionRecallDisplay.from_predictions(TSOUT['y'], TSOUT['phglm'], ax=ax[1,2]);
display.ax_.set_title('PrecisionRecallDisplay glm on TS');


# In[23]:


# 직접 그리기 roc_curve, precision_recall_curve는 좌표를 반환함
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4), constrained_layout=True)
# 기본마진 fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2) 

fpr, tpr, thres = roc_curve(TROUT['y'], TROUT['phglm'], pos_label=1) 
aucroc = roc_auc_score(TROUT['y'], TROUT['phglm'])
ax[0].plot(fpr, tpr, 'b', label='AUC=%.2f'%aucroc);
ax[0].legend(loc='lower right')
ax[0].plot([0,1],[0,1],'r--')
ax[0].set_xlim([-0.05,1.05]) 
ax[0].set_ylim([-0.05,1.05]) 
ax[0].set_xlabel('FPR(=1-spec)')
ax[0].set_ylabel('TPR(=sens)')
ax[0].set_title('roc_curve glm on TR')

# 직접 그리기 PR curve
prec, rec, thre = precision_recall_curve(TROUT['y'], TROUT['phglm'], pos_label=1) 
aucpr = average_precision_score(TROUT['y'], TROUT['phglm'])

ax[1].plot(rec, prec, 'b', label='AUC(PR)=%.2f'%aucpr);
ax[1].legend(loc='lower right')
ax[1].set_xlim([-0.05,1.05]) 
ax[1].set_ylim([-0.05,1.05]) 
ax[1].set_xlabel('recall')
ax[1].set_ylabel('precision')
ax[1].set_title('precision_recall_curve glm on TR')


# * cutoff 조정

# In[24]:


# Youden's J = sens + spec - 1  where spec=1-fpr
# closest.topleft from FPR, TPR, THRES
CUTOFF = pd.DataFrame( {'youden': thres[np.argmax(tpr+(1-fpr)-1)],
                        'topleft': thres[np.argmin(np.power(1-tpr, 2)+np.power(fpr,2))]}, index=['glm'])
CUTOFF


# In[25]:


yh = np.where(TROUT['phglm']>CUTOFF.loc['glm', 'youden'], 1, 0)
metbin(TROUT['y'], yh, modelname='TRglm')


# ### OPenet
# * ElasticNet: L1, L2 를 결합한 Logit: 
# * `penalty='elasticnet', solver='saga'` 지정
# * optuna로 `C, l1_ratio` 튜닝
# 
# ```
# ! pip install ipywidgets
# ! jupyter nbextension enable --py widgetsnbextension
# ```

# In[49]:


import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore') 
# optuna.visualization.is_available()  # plotly 사용가능한지 확인


# In[50]:


# 목적함수 정의 
# trial.suggest_[float,uniform,loguniform,discrete_uniform]
def objenet(trial):
    C        = trial.suggest_float('C', 0.001, 10)
    l1_ratio = trial.suggest_float('l1_ratio', 0.00001, 0.99999)
    
    PEenet = Pipeline([ ('PP', PP), 
                        ('enet', LogisticRegression(penalty='elasticnet', 
                                                    max_iter=1000, 
                                                    solver='saga',
                                                    C = C,
                                                    l1_ratio=l1_ratio)) ])
    cvacc = cross_val_score(PEenet, TRX, TRy, cv=SKF, scoring='accuracy') #, scoring='neg_root_mean_squared_error')
    return cvacc.mean()

# Execute optuna and set hyperparameters
STenet = optuna.create_study(direction='maximize')
get_ipython().run_line_magic('time', 'STenet.optimize(objenet, n_trials=10)  # 23분 소요')


# In[51]:


# 목적함수값, 튜닝된 모수값
STenet.best_value, STenet.best_params


# In[52]:


# Trial vs obj(acc)
from optuna.visualization import plot_optimization_history, plot_slice, plot_contour, plot_parallel_coordinate
plot_optimization_history(STenet)


# In[53]:


# 모수별 ojb(acc) 값
plot_slice(STenet)


# In[54]:


plot_contour(STenet, params=['C', 'l1_ratio'], target_name='Accuracy')


# In[55]:


plot_parallel_coordinate(STenet, target_name='Accuracy')


# In[56]:


STenet.trials_dataframe().sort_values(by='value', ascending=False).head()  


# In[57]:


OPenet = Pipeline([ ('PP', PP), 
                    ('enet', LogisticRegression(penalty='elasticnet', 
                                                max_iter=1000, 
                                                solver='saga',
                                                C = STenet.best_params['C'],
                                                l1_ratio=STenet.best_params['l1_ratio'])) ])

OPenet.fit(TRX ,TRy)
OPenet.score(TRX ,TRy), OPenet.score(TSX ,TSy)


# In[58]:


TROUT['yhenet'] = OPenet.predict(TRX)              
TROUT['phenet'] = OPenet.predict_proba(TRX)[:,1]   
TSOUT['yhenet'] = OPenet.predict(TSX) 
TSOUT['phenet'] = OPenet.predict_proba(TSX)[:,1] 
TSOUT.head()



# In[62]:
# 다수 지표 한번에 계산
CV = pd.DataFrame(cross_validate(OPenet, TRX, TRy, scoring=scr, cv=SKF))  # dict반환. 5-fold CV결과
CV = CV.filter(regex='test').mean()   # dplyr::filter와 다름 
CV = CV.to_frame(name='enet').T
CVMET = pd.concat([CVMET, CV], axis=0)
CVMET


# In[63]:


MET   = metbin(TROUT['y'], TROUT['yhenet'], TROUT['phenet'], modelname='TRenet')
TRMET = pd.concat([TRMET, MET], axis=0)

MET   = metbin(TSOUT['y'], TSOUT['yhenet'], TSOUT['phenet'], modelname='TSenet')
TSMET = pd.concat([TSMET, MET], axis=0)



# In[65]:


fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10,5), constrained_layout=True)
# 기본마진 fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2) 

display = ConfusionMatrixDisplay.from_predictions(TROUT['y'], TROUT['yhenet'], ax=ax[0,0])
display.ax_.set_title('ConfusionMatrixDisplay(TR)');


# Estimator를 이용하려면 RocCurveDisplay.from_estimator(estimator, X, y, ..)
# ph이 있으면 RocCurveDisplay.from_predictions(y, ph, ...)
from sklearn.metrics import RocCurveDisplay
display = RocCurveDisplay.from_predictions(TROUT['y'], TROUT['phenet'], ax=ax[0,1]);
display.ax_.set_title('RocCurveDispaly enet on TR');


# Estimator를 이용하려면 PrecisionRecallDisplay.from_estimator(E, X, y, ..)
# ph이 있으면 PrecisionRecallDisplay.from_predicitions(y, ph, ..)
from sklearn.metrics import PrecisionRecallDisplay
display = PrecisionRecallDisplay.from_predictions(TROUT['y'], TROUT['phenet'], ax=ax[0,2]);
display.ax_.set_title('PrecisionRecallDisplay enet on TR');


display = ConfusionMatrixDisplay.from_predictions(TSOUT['y'], TSOUT['yhenet'], ax=ax[1,0])
display.ax_.set_title('ConfusionMatrixDisplay enet on TS');


# Estimator를 이용하려면 RocCurveDisplay.from_estimator(estimator, X, y, ..)
# ph이 있으면 RocCurveDisplay.from_predictions(y, ph, ...)
from sklearn.metrics import RocCurveDisplay
display = RocCurveDisplay.from_predictions(TSOUT['y'], TSOUT['phenet'], ax=ax[1,1]);
display.ax_.set_title('RocCurveDispaly enet on TS');


# Estimator를 이용하려면 PrecisionRecallDisplay.from_estimator(E, X, y, ..)
# ph이 있으면 PrecisionRecallDisplay.from_predicitions(y, ph, ..)
from sklearn.metrics import PrecisionRecallDisplay
display = PrecisionRecallDisplay.from_predictions(TSOUT['y'], TSOUT['phenet'], ax=ax[1,2]);
display.ax_.set_title('PrecisionRecallDisplay enet on TS');


# In[66]:


fpr, tpr, thres = roc_curve(TROUT['y'], TROUT['phenet'], pos_label=1) 
tmp = pd.DataFrame( {'youden':  thres[np.argmax(tpr+(1-fpr)-1)], 
                     'topleft': thres[np.argmin(np.power(1-tpr, 2)+np.power(fpr,2))]}, index=['enet'])
CUTOFF = pd.concat([CUTOFF, tmp], axis=0)
CUTOFF




## 컷오프 이용하기
### Youden J를 이용한 컷오프로  yhglm 생성
### 평가지표 변화 (TRMET, TSMET) 계산
###  오분류표 생성/시각화
## class_weight='balanced' ridge 모형생성
###  optuna로 튜닝한 모형 생성
###  평가지표 변화
###  오분류표 생성시각화


