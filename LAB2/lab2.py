import numpy as np
import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

#from pandas import DataFrame
#df = DataFrame.from_csv("train.tsv", sep="\t")
#print df.shape

## Reads the training file
with open('train.tsv') as f:
    df = pd.read_table(f, sep='\t',header=None)

## Prints some information about the data frame
print df.shape
print df.head()

ytrain = df[df.columns[0]]
xtrain = df[df.columns[1]]

## The given tokenizer
tokenize_re = re.compile(r'''
                         \d+[:\.]\d+
                         |(https?://)?(\w+\.)(\w{2,})+([\w/]+)
                         |[@\#]?\w+(?:[-']\w+)*
                         |[^a-zA-Z0-9 ]+''',
                         re.VERBOSE)

def tokenize(text):
    return [ m.group() for m in tokenize_re.finditer(text) ]

## Playing with the vectorizer
vec = CountVectorizer(tokenizer = tokenize)
X = vec.fit_transform(xtrain)
print X.toarray()
##print vec.get_feature_names()[100:130]
message1 = xtrain[1]
message1_trans = vec.transform([message1])
print message1_trans
print vec.get_feature_names()[2038]

## Playing with the feature selection function

feature_score = f_classif(X,ytrain)[0]
for score, fname in sorted(zip(feature_score,vec.get_feature_names()),reverse=True)[:100]:
    print(fname, score)


## Dummy classifier
pipeline_dummy = make_pipeline(
CountVectorizer(tokenizer = tokenize),
    SelectPercentile(percentile=10),
    DummyClassifier()
)


print(cross_validate(pipeline_dummy, xtrain, ytrain))



############################ Decision Tree
pipeline_tree = make_pipeline(
CountVectorizer(tokenizer = tokenize),
    SelectPercentile(percentile=10),
    DecisionTreeClassifier(max_depth=20)
)

print(cross_validate(pipeline_tree, xtrain, ytrain))
## Default: 0.60
## MD = 50: 0.61
## MD = 40: 0.61
## MD = 30: 0.62
## MD = 20: 0.61

max_d = np.random.random_integers(10,200,(20))
res = np.zeros(20)
for x in np.arange(0,10):
    pipeline_tree = make_pipeline(
        CountVectorizer(tokenizer = tokenize),
        SelectPercentile(percentile=10),
        DecisionTreeClassifier(max_depth=max_d[x])
    )
    cv = cross_validate(pipeline_RF, xtrain, ytrain)
    res[x] = cv['test_score'].mean()

print(max_d[np.argmax(res)])
print(res.max())

## Max deapth = 183, score = 0.66

    




########################### Random Forest
pipeline_RF = make_pipeline(
CountVectorizer(tokenizer = tokenize),
    SelectPercentile(percentile=10),
    RandomForestClassifier()
)

print(cross_validate(pipeline_RF, xtrain, ytrain))
## Default: 0.62

n_est = np.random.random_integers(1,100,(10))
max_d = np.random.random_integers(100,200,(10))
res = np.zeros(10)
param = zip(n_est,max_d)
for x in np.arange(0,10):
    pipeline_RF = make_pipeline(
        CountVectorizer(tokenizer = tokenize),
        SelectPercentile(percentile=10),
        RandomForestClassifier(n_estimators = param[x][0],
                               max_depth=param[x][1])
    )
    cv = cross_validate(pipeline_RF, xtrain, ytrain)
    res[x] = cv['test_score'].mean()

print(param[np.argmax(res)])    

## number of trees: 59, max depth = 135, score: 0.66


## Gradient boosted classifier
pipeline_GBC = make_pipeline(
CountVectorizer(tokenizer = tokenize),
    SelectPercentile(percentile=10),
    GradientBoostingClassifier()
)

print(cross_validate(pipeline_GBC, xtrain, ytrain))

## Logistic Regression
pipeline_LR = make_pipeline(
CountVectorizer(tokenizer = tokenize),
    SelectPercentile(percentile=10),
    LogisticRegression()
)

print(cross_validate(pipeline_LR, xtrain, ytrain))

## Linear SVC
pipeline_SVC = make_pipeline(
CountVectorizer(tokenizer = tokenize),
    SelectPercentile(percentile=10),
    LinearSVC()
)

print(cross_validate(pipeline_SVC, xtrain, ytrain))

## Neural Network
pipeline_NN = make_pipeline(
CountVectorizer(tokenizer = tokenize),
    SelectPercentile(percentile=10),
    MLPClassifier(hidden_layer_sizes =(200))
)

print(cross_validate(pipeline_NN, xtrain, ytrain))
## Default: 0.63
## 100 100: 0.63
## 50 50 50: 0.63
## 100 100 100 100: 0.63
## 200: 0.64
## Wider is better but takes longer
