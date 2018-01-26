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


pipeline = make_pipeline(
CountVectorizer(tokenizer = tokenize),
    SelectPercentile(percentile=10),
    DecisionTreeClassifier()
)


print(cross_validate(pipeline, xtrain, ytrain))
