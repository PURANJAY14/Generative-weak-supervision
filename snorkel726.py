import re
import glob

import os
from snorkel.analysis import get_label_buckets
from snorkel.labeling import labeling_function
from snorkel.labeling import LFAnalysis
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LabelingFunction
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.labeling.lf.nlp import nlp_labeling_function
from snorkel.preprocess import preprocessor
from snorkel.preprocess.nlp import SpacyPreprocessor

from snorkel.utils import probs_to_preds

from textblob import TextBlob

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import matplotlib.pyplot as plt
from snorkel.analysis import metric_score

def analyze(v,out):
    dict={}
    score=[-1,1]
    v=np.array(v)
    out=np.array(out)
    print(v,len(out))
    for i in range(len(out)):
        if v[i] in dict.keys():
            dict[v[i]]+=score[out[i]]
        else:
            dict[v[i]]=((int)(0))

    print(OrderedDict(sorted(dict.items())))

print(os.getcwd())
df_temp= pd.read_csv('data/train.csv')

df=pd.get_dummies(df_temp)
print(df_temp)
df.rename(columns={'hours-per-week':'hours_per_week'},inplace=True)
df.rename(columns={'income_>50K':'income_50k'},inplace=True)
df.rename(columns={'educational-num':'education'},inplace=True)
df.rename(columns={'capital-gain':'capital_gain'},inplace=True)
df.rename(columns={'capital-loss':'capital_loss'},inplace=True)
df_train=df.iloc[1:40000,:]
df_test = df.iloc[40000:,:]


# for i in df_train.columns:
#     print(i)
#     if i!='fnlwgt':
#         analyze(df_train[i],df_train.income_50k)

df_train.drop('income_50k',axis=1)
df_test.drop('income_50k',axis=1)

print(df_train,df_test.columns)



Y_test=df.income_50k.values[40000:]


@labeling_function()
def check(x):
    # print(int(x.hours_per_week),x)
    return 0 if int(x.hours_per_week)<40 else -1

@labeling_function()
def check2(x):
    # print(int(x.hours_per_week),x)
    return 0 if x.education<14 else -1

@labeling_function()
def check3(x):
    # print(int(x.hours_per_week),x)
    return 1 if x.capital_gain>=8000 else -1

@labeling_function()
def check4(x):
    # print(int(x.hours_per_week),x)
    return 0 if x.capital_loss>=3500 else -1




lfs = [check,check2,check3,check4]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)
L_test = applier.apply(df=df_test)

coverage_check= (L_train != -1).mean(axis=0)

print(coverage_check)
# print(f"check coverage: {coverage_check * 100:.1f}%")

print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())



majority_model = MajorityLabelVoter()
preds_train = majority_model.predict(L=L_train)


label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)
pred_test = label_model.predict(L_test)

acc = metric_score(Y_test,pred_test, probs=None, metric="accuracy")
print(acc)

majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]
label_model_acc = label_model.score(L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]


probs_train = label_model.predict_proba(L=L_train)

df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=df_train, y=probs_train, L=L_train
)


X_train=np.array(df_train_filtered)
X_test=np.array(df_test)
preds_train_filtered = probs_to_preds(probs=probs_train_filtered)


sklearn_model = LogisticRegression(C=1e3, solver="liblinear")
sklearn_model.fit(X=X_train, y=preds_train_filtered)

print(sklearn_model.score(X=X_test, y=Y_test))




