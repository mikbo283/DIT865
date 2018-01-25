import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re

from pandas import DataFrame
df = DataFrame.from_csv("LAB2/train.tsv", sep="\t")


print df.shape


with open('LAB2/train.tsv') as f:
    table = pd.read_table(f, sep='\t')


print table.shape

print table.head(5)

print table[table.columns[0]]


tokenize_re = re.compile(r'''
                         \d+[:\.]\d+
                         |(https?://)?(\w+\.)(\w{2,})+([\w/]+)
                         |[@\#]?\w+(?:[-']\w+)*
                         |[^a-zA-Z0-9 ]+''',
                         re.VERBOSE)

def tokenize(text):
    return [ m.group() for m in tokenize_re.finditer(text) ]


vec = CountVectorizer(tokenizer = tokenize)

corpus = ["This is is.", "gs gs gs is"]

X = vec.fit_transform(table)

print X.toarray()
print vec.get_feature_names()






X = [['example', 'text'],
     ['another', 'text']]

vec = CountVectorizer(preprocessor = lambda x: x,
                      tokenizer = lambda x: x)
Xe = vec.fit_transform(X)
print(Xe.toarray())

print(vec.get_feature_names())
