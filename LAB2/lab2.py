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
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import f_classif
from sklearn.feature_extraction import DictVectorizer


##from pandas import DataFrame
##df = DataFrame.from_csv("train.tsv", sep="\t")
##print df.shape

## Reads the training file
with open('train.tsv') as f:
    df = pd.read_table(f, sep='\t',header=None)



""""
import pandas as pd
from pandas import DataFrame
df = DataFrame.from_csv("train.tsv", sep="\t")
df = pd.read_table("train.tsv", sep="\t", header=None)
df_y = df[0]
df_x = df[1]
dev = DataFrame.from_csv("dev.tsv", sep="\t")
dev = pd.read_table("train.tsv", sep="\t", header=None)
dev_x = dev[1]
dev_y = dev[0]
"""""
    
    
## Prints some information about the data frame
print df.shape
print df.head()

ytrain = df[df.columns[0]]
xtrain = df[df.columns[1]]
""""
xtrain = df[1]
ytrain = df[0]
"""""
## Maake lowercase
xtrain[:] = [x.lower() for x in xtrain]


## Reads the development file
with open('dev.tsv') as f:
    dfdev = pd.read_table(f, sep='\t',header=None)

## Prints some information about the data frame
print dfdev.shape
print dfdev.head()

ydev = dfdev[dfdev.columns[0]]
xdev = dfdev[dfdev.columns[1]]
""""
ydev = dfdev[0]
xdev = dfdev[1]
"""""
##Make lowercase
xdev[:] = [x.lower() for x in xdev]



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
print vec.get_feature_names()[5062]

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


####################################################################
############################ Decision Tree #########################
####################################################################

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

## PREDICTION: TREE

pipeline_tree_best = make_pipeline(
    CountVectorizer(tokenizer = tokenize),
    StandardScaler(with_mean=False),
    SelectPercentile(percentile=10),
    DecisionTreeClassifier(max_depth=183)
)

model_tree = pipeline_tree_best.fit(xtrain,ytrain)

Yguess = model_tree.predict(xdev)
print(accuracy_score(ydev, Yguess))
## Score: 0.579    

####################################################################
########################### Random Forest ##########################
####################################################################

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

## Predict RF

pipeline_RF_best = make_pipeline(
CountVectorizer(tokenizer = tokenize),
    StandardScaler(with_mean=False),
    SelectPercentile(percentile=10),
    RandomForestClassifier(n_estimators = 59, max_depth = 135)
)

model_RF = pipeline_RF_best.fit(xtrain,ytrain)

Yguess = model_RF.predict(xdev)
print(accuracy_score(ydev, Yguess))

## Score: 0.61

####################################################################
############### Gradient boosted classifier ########################
####################################################################

pipeline_GBC = make_pipeline(
CountVectorizer(tokenizer = tokenize),
    SelectPercentile(percentile=10),
    GradientBoostingClassifier()
)

print(cross_validate(pipeline_GBC, xtrain, ytrain))
## Default: 0.66

rates = np.random.rand(10)
res = np.zeros(10)
for x in np.arange(0,1):
    pipeline_GBC = make_pipeline(
        CountVectorizer(tokenizer = tokenize),
        SelectPercentile(percentile=10),
        GradientBoostingClassifier(learning_rate = rates[x])
    )
    cv = cross_validate(pipeline_GBC, xtrain, ytrain)
    res[x] = cv['test_score'].mean()

print(rates[np.argmax(res)])
print(res.max())
## learning rate = 0.427: 0.68

### Predict GBC

pipeline_GBC_best = make_pipeline(
    CountVectorizer(tokenizer = tokenize),
    StandardScaler(with_mean=False),
    SelectPercentile(percentile=10),
    GradientBoostingClassifier()
)

model_GBC = pipeline_GBC_best.fit(df_x,df_y)

Yguess = model_GBC.predict(dev_x)
print(accuracy_score(dev_x, Yguess))

## Score: 0.649

####################################################################
##################### Logistic Regression ##########################
####################################################################

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
## Default: 0.685

#### Predict LR

pipeline_LR_best = make_pipeline(
CountVectorizer(tokenizer = tokenize),
    StandardScaler(with_mean=False),
    SelectPercentile(percentile=10),
    LogisticRegression()
)

model_LR = pipeline_LR_best.fit(xtrain,ytrain)

Yguess = model_LR.predict(xdev)
print(accuracy_score(ydev, Yguess))

##Score: 0.636

####################################################################
###################### Neural Network ##############################
####################################################################
pipeline_NN = make_pipeline(
CountVectorizer(tokenizer = tokenize),
    SelectPercentile(percentile=10),
    MLPClassifier(hidden_layer_sizes =(300))
)

print(cross_validate(pipeline_NN, xtrain, ytrain))
## Default: 0.63
## 100 100: 0.63
## 50 50 50: 0.63
## 100 100 100 100: 0.63
## 200: 0.64
## 300: 0.645
## Wider is better but takes longer

pipeline_NN_best = make_pipeline(
    CountVectorizer(tokenizer = tokenize),
    StandardScaler(with_mean=False),
    SelectPercentile(percentile=10),
    MLPClassifier(hidden_layer_sizes =(300))
)

model_NN = pipeline_NN_best.fit(xtrain,ytrain)

Yguess = model_NN.predict(xdev)
print(accuracy_score(ydev, Yguess))

## Score: 0.636

################# Cluster ########

with open('50mpaths2') as f:
    df = pd.read_table(f, sep = '\t', converters = {'Code': lambda x: str(x)})

df2 = df[["Code", "Key"]]
D = dict(zip(df2.Key,df2.Code))
    
## A function that maps a string to a word cluster (from sourcr provided in lab instructions)    
def cluster_word(string):
    if D.get(string) == None:
        return string
    else:
        return D.get(string)

## A tokenizer based on a comibation of the cluster_word function and
## the tokenizer provided in the lab instructions
def cluster_text(text):
    return [cluster_word(string) for string in tokenize(text)]

#def cluster_text(text):
#    return [cluster_word(string) for string in text.split()]


#################################################################
#################################################################
############# Repeat the learning with new tokenizer ############
#################################################################
#################################################################

####################################################################
############################ Decision Tree #########################
####################################################################

pipeline_tree = make_pipeline(
CountVectorizer(tokenizer = cluster_text),
    SelectPercentile(percentile=10),
    DecisionTreeClassifier()
)

print(cross_validate(pipeline_tree, xtrain, ytrain))
## Default: 0.59

max_d = np.random.random_integers(10,200,(30))
res = np.zeros(20)
for x in np.arange(0,10):
    pipeline_tree = make_pipeline(
        CountVectorizer(tokenizer = cluster_text),
        SelectPercentile(percentile=10),
        DecisionTreeClassifier(max_depth=max_d[x])
    )
    cv = cross_validate(pipeline_tree, xtrain, ytrain)
    res[x] = cv['test_score'].mean()

print(max_d[np.argmax(res)])
print(res.max())

## Max deapth = 42, score = 0.62

## PREDICTION: TREE

pipeline_tree_best = make_pipeline(
    CountVectorizer(tokenizer = cluster_text),
    StandardScaler(with_mean=False),
    SelectPercentile(percentile=10),
    DecisionTreeClassifier(max_depth=42)
)

model_tree = pipeline_tree_best.fit(xtrain,ytrain)

Yguess = model_tree.predict(xdev)
print(accuracy_score(ydev, Yguess))
## Score: 0.596    

####################################################################
########################### Random Forest ##########################
####################################################################

pipeline_RF = make_pipeline(
CountVectorizer(tokenizer = cluster_text),
    SelectPercentile(percentile=10),
    RandomForestClassifier()
)

print(cross_validate(pipeline_RF, xtrain, ytrain))
## Default: 0.64

n_est = np.random.random_integers(1,100,(20))
max_d = np.random.random_integers(100,200,(20))
res = np.zeros(10)
param = zip(n_est,max_d)
for x in np.arange(0,10):
    pipeline_RF = make_pipeline(
        CountVectorizer(tokenizer = cluster_text),
        SelectPercentile(percentile=10),
        RandomForestClassifier(n_estimators = param[x][0],
                               max_depth=param[x][1])
    )
    cv = cross_validate(pipeline_RF, xtrain, ytrain)
    res[x] = cv['test_score'].mean()

print(param[np.argmax(res)])    
print(res.max())

## number of trees: 96, max depth = 171, score: 0.677

## Predict RF

pipeline_RF_best = make_pipeline(
CountVectorizer(tokenizer = cluster_text),
    StandardScaler(with_mean=False),
    SelectPercentile(percentile=10),
    RandomForestClassifier(n_estimators = 96, max_depth = 171)
)

model_RF = pipeline_RF_best.fit(xtrain,ytrain)

Yguess = model_RF.predict(xdev)
print(accuracy_score(ydev, Yguess))

## Score: 0.630

####################################################################
############### Gradient boosted classifier ########################
####################################################################

pipeline_GBC = make_pipeline(
CountVectorizer(tokenizer = cluster_text),
    SelectPercentile(percentile=10),
    GradientBoostingClassifier()
)

print(cross_validate(pipeline_GBC, xtrain, ytrain))
## Default: 0.675

rates = np.random.rand(20)
res = np.zeros(20)
for x in np.arange(0,1):
    pipeline_GBC = make_pipeline(
        CountVectorizer(tokenizer = cluster_text),
        SelectPercentile(percentile=10),
        GradientBoostingClassifier(learning_rate = rates[x])
    )
    cv = cross_validate(pipeline_GBC, xtrain, ytrain)
    res[x] = cv['test_score'].mean()

print(rates[np.argmax(res)])
print(res.max())
## learning rate = 0.585: 0.683

### Predict GBC

pipeline_GBC_best = make_pipeline(
CountVectorizer(tokenizer = cluster_text),
    StandardScaler(with_mean=False),
    SelectPercentile(percentile=10),
    GradientBoostingClassifier(learning_rate = 0.5853)
)

model_GBC = pipeline_GBC_best.fit(xtrain,ytrain)

Yguess = model_GBC.predict(xdev)
print(accuracy_score(ydev, Yguess))

## Score: 0.659

####################################################################
##################### Logistic Regression ##########################
####################################################################

pipeline_LR = make_pipeline(
CountVectorizer(tokenizer = cluster_text),
    SelectPercentile(percentile=10),
    LogisticRegression()
)

print(cross_validate(pipeline_LR, xtrain, ytrain))
## Default: 0.69

pipeline_LR_best = make_pipeline(
CountVectorizer(tokenizer = cluster_text),
    StandardScaler(with_mean=False),
    SelectPercentile(percentile=10),
    LogisticRegression()
)

model_LR = pipeline_LR_best.fit(xtrain,ytrain)

Yguess = model_LR.predict(xdev)
print(accuracy_score(ydev, Yguess))
## Score: 0.666

###############################################################
#################### Linear SVC ###############################
###############################################################

## Linear SVC
pipeline_SVC = make_pipeline(
CountVectorizer(tokenizer = cluster_text),
    SelectPercentile(percentile=10),
    LinearSVC()
)

print(cross_validate(pipeline_SVC, xtrain, ytrain))
## Default: 0.688

#### Predict SVC

pipeline_SVC_best = make_pipeline(
CountVectorizer(tokenizer = cluster_text),
    StandardScaler(with_mean=False),
    SelectPercentile(percentile=10),
    LinearSVC()
)

model_SVC = pipeline_SVC_best.fit(xtrain,ytrain)

Yguess = model_SVC.predict(xdev)
print(accuracy_score(ydev, Yguess))

##Score: 0.652

####################################################################
###################### Neural Network ##############################
####################################################################
pipeline_NN = make_pipeline(
    CountVectorizer(tokenizer = cluster_text),
    SelectPercentile(percentile=10),
    MLPClassifier(hidden_layer_sizes =(100, 100))
)

print(cross_validate(pipeline_NN, xtrain, ytrain))
## Default: 0.66
## 200: 0.65
## 100 100: 0.665
## 100 100 100: 0.63
## 200 200: 0.64
## 50: 0.651

pipeline_NN_best = make_pipeline(
    CountVectorizer(tokenizer = cluster_text),
    StandardScaler(with_mean=False),
    SelectPercentile(percentile=10),
    MLPClassifier(hidden_layer_sizes =(100))
)

model_NN = pipeline_NN_best.fit(xtrain,ytrain)

Yguess = model_NN.predict(xdev)
print(accuracy_score(ydev, Yguess))

## Score: 0.633


################################################################
################################################################
################## Replace code with most freq word ############
################################################################
################################################################

unique_code = df.Code.unique()
corr_word = df.Code.unique()

for x in np.arange(0,unique_code.size):
    TT = (df.loc[df['Code']==unique_code[x]])
    corr_word[x] = df.iat[TT['Freq'].idxmax(),1]
    
corr_dict = dict(zip(unique_code,corr_word))    

## A function that corrects a word according to the cluster privided in the assignment   
def correct_word(string):
    if D.get(string) == None:
        return str(string)
    else:
        return str(corr_dict.get(D.get(string)))


    

## A tokenizer based on a comibation of the cluster_word function and the tokenizer provided in the lab instructions
def correct_text(text):
    return [correct_word(string) for string in tokenize(text)]


##### What are the most important features?
vec = CountVectorizer(tokenizer = correct_text)
X = vec.fit_transform(xtrain)

feature_scores = f_classif(X, ytrain)[0]

for score, fname in sorted(zip(feature_scores, vec.get_feature_names()), reverse=True)[:100]:
    print(fname, score)


#################################################################
#################################################################
############ Repeat the learning with replaced words ############
#################################################################
#################################################################

####################################################################
############################ Decision Tree #########################
####################################################################

pipeline_tree = make_pipeline(
CountVectorizer(tokenizer = correct_text),
    SelectPercentile(percentile=10),
    DecisionTreeClassifier()
)

print(cross_validate(pipeline_tree, xtrain, ytrain))
## Default: 0.59

max_d = np.random.random_integers(10,200,(30))
res = np.zeros(20)
for x in np.arange(0,10):
    pipeline_tree = make_pipeline(
        CountVectorizer(tokenizer = correct_text),
        SelectPercentile(percentile=5),
        DecisionTreeClassifier(max_depth=max_d[x])
    )
    cv = cross_validate(pipeline_tree, xtrain, ytrain)
    res[x] = cv['test_score'].mean()

print(max_d[np.argmax(res)])
print(res.max())

## Max deapth = 25, score = 0.62

## PREDICTION: TREE

pipeline_tree_best = make_pipeline(
    CountVectorizer(tokenizer = correct_text),
    StandardScaler(with_mean=False),
    SelectPercentile(percentile=5),
    DecisionTreeClassifier(max_depth=25)
)

model_tree = pipeline_tree_best.fit(xtrain,ytrain)

Yguess = model_tree.predict(xdev)
print(accuracy_score(ydev, Yguess))
## Score: 0.59    

####################################################################
########################### Random Forest ##########################
####################################################################

pipeline_RF = make_pipeline(
CountVectorizer(tokenizer = correct_text),
    SelectPercentile(percentile=5),
    RandomForestClassifier()
)

print(cross_validate(pipeline_RF, xtrain, ytrain))
## Default: 0.64

n_est = np.random.random_integers(1,100,(20))
max_d = np.random.random_integers(100,200,(20))
res = np.zeros(10)
param = zip(n_est,max_d)
for x in np.arange(0,10):
    pipeline_RF = make_pipeline(
        CountVectorizer(tokenizer = correct_text),
        SelectPercentile(percentile=5),
        RandomForestClassifier(n_estimators = param[x][0],
                               max_depth=param[x][1])
    )
    cv = cross_validate(pipeline_RF, xtrain, ytrain)
    res[x] = cv['test_score'].mean()

print(param[np.argmax(res)])    
print(res.max())

## number of trees: 64, max depth = 133, score: 0.677

## Predict RF

pipeline_RF_best = make_pipeline(
CountVectorizer(tokenizer = correct_text),
    StandardScaler(with_mean=False),
    SelectPercentile(percentile=5),
    RandomForestClassifier(n_estimators = 64, max_depth = 133)
)

model_RF = pipeline_RF_best.fit(xtrain,ytrain)

Yguess = model_RF.predict(xdev)
print(accuracy_score(ydev, Yguess))

## Score: 0.636

####################################################################
############### Gradient boosted classifier ########################
####################################################################

pipeline_GBC = make_pipeline(
CountVectorizer(tokenizer = correct_text),
    SelectPercentile(percentile=5),
    GradientBoostingClassifier()
)

print(cross_validate(pipeline_GBC, xtrain, ytrain))
## Default: 0.675

rates = np.random.rand(20)
res = np.zeros(20)
for x in np.arange(0,1):
    pipeline_GBC = make_pipeline(
        CountVectorizer(tokenizer = correct_text),
        SelectPercentile(percentile=5),
        GradientBoostingClassifier(learning_rate = rates[x])
    )
    cv = cross_validate(pipeline_GBC, xtrain, ytrain)
    res[x] = cv['test_score'].mean()

print(rates[np.argmax(res)])
print(res.max())
## learning rate = 0.123: 0.682

### Predict GBC

pipeline_GBC_best = make_pipeline(
    CountVectorizer(tokenizer = correct_text),
    StandardScaler(with_mean=False),
    SelectPercentile(percentile=5),
    GradientBoostingClassifier(learning_rate = 0.123)
)

model_GBC = pipeline_GBC_best.fit(xtrain,ytrain)

Yguess = model_GBC.predict(xdev)
print(accuracy_score(ydev, Yguess))

## Score: 0.638

####################################################################
##################### Logistic Regression ##########################
####################################################################

pipeline_LR = make_pipeline(
CountVectorizer(tokenizer = correct_text),
    SelectPercentile(percentile=5),
    LogisticRegression()
)

print(cross_validate(pipeline_LR, xtrain, ytrain))
## Default: 0.695

pipeline_LR_best = make_pipeline(
CountVectorizer(tokenizer = correct_text),
    StandardScaler(with_mean=False),
    SelectPercentile(percentile=5),
    LogisticRegression()
)

model_LR = pipeline_LR_best.fit(xtrain,ytrain)

Yguess = model_LR.predict(xdev)
print(accuracy_score(ydev, Yguess))
## Score: 0.674

###############################################################
#################### Linear SVC ###############################
###############################################################

## Linear SVC
pipeline_SVC = make_pipeline(
    CountVectorizer(tokenizer = correct_text),
    SelectPercentile(percentile=5),
    LinearSVC()
)

print(cross_validate(pipeline_SVC, xtrain, ytrain))
## Default: 0.694

#### Predict SVC

pipeline_SVC_best = make_pipeline(
CountVectorizer(tokenizer = correct_text),
    StandardScaler(with_mean=False),
    SelectPercentile(percentile=5),
    LinearSVC()
)

model_SVC = pipeline_SVC_best.fit(xtrain,ytrain)

Yguess = model_SVC.predict(xdev)
print(accuracy_score(ydev, Yguess))

##Score: 0.668

####################################################################
###################### Neural Network ##############################
####################################################################
pipeline_NN = make_pipeline(
    CountVectorizer(tokenizer = correct_text),
    SelectPercentile(percentile=10),
    MLPClassifier(hidden_layer_sizes =(100, 100))
)

print(cross_validate(pipeline_NN, xtrain, ytrain))


pipeline_NN_best = make_pipeline(
    CountVectorizer(tokenizer = correct_text),
    StandardScaler(with_mean=False),
    SelectPercentile(percentile=10),
    MLPClassifier(hidden_layer_sizes =(100))
)

model_NN = pipeline_NN_best.fit(xtrain,ytrain)

Yguess = model_NN.predict(xdev)
print(accuracy_score(ydev, Yguess))













    
## Sentiment lexicon

with open('SemEval2015-English-Twitter-Lexicon/SemEval2015-English-Twitter-Lexicon.txt') as f:
    sentiment_df = pd.read_table(f, sep = '\t')

D_sentiment = dict(zip(sentiment_df.Key,sentiment_df.Value))

print [D_sentiment.get(string) for string in tokenize(xtrain[0])]

def sentiment(string):
    if D_sentiment.get(string) == None:
        return 0
    else:
        return D_sentiment.get(string)

def sentify(text):
    v = [sentiment(string) for string in make_interactions(text)]
    return sum(v)

def get_sentiment(xdf):
    return np.array([sentify(text) for text in xdf]).reshape(-1, 1)






## A function that corrects a word according to the cluster privided in the assignment   
def correct_word(string):
    if D.get(string) == None:
        return str(string)
    else:
        return str(corr_dict.get(D.get(string)))

## A tokenizer based on a comibation of the cluster_word function and the tokenizer provided in the lab instructions
def correct_text(text):
    return [correct_word(string) for string in tokenize(text)]





































	


import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import FunctionTransformer

def get_text_length(x):
    return np.array([len(t) for t in x]).reshape(-1, 1)

classifier = Pipeline([
    ('features', FeatureUnion([
        ('text', Pipeline([
            ('vectorizer', CountVectorizer(tokenizer=correct_text)),
            ('feat', SelectPercentile(percentile=5))
        ])),
        ('score', Pipeline([
            ('sentiment', FunctionTransformer(get_sentiment, validate=False)),
        ]))        
    ])),
    ('clf', OneVsRestClassifier(LogisticRegression()))])

model = classifier.fit(xtrain,ytrain)

Yguess = model.predict(xdev)
print(accuracy_score(ydev, Yguess))

## RandomForest: 0.615
## LinearSVC: 0.678
## LogisticRegression: 0.676

from sklearn.metrics import confusion_matrix
print confusion_matrix(ydev,Yguess)


########################## Word Pairs ########################


    for first, second in zip(correct_text(xtrain[0]), correct_text(xtrain[0])[1:]):
        print first, second

def make_interactions(list1):
    list2 = correct_text(list1)
    new_list = list2 + [' '.join(x) for x in zip(list2,list2[1:])]
    return new_list
