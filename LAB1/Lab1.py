import sklearn
import numpy as np

## A function that reads the names and types of the attributes in a file
def read_feature_descriptions(filename): 
    names = []
    types = []
    with open(filename) as f:
        for l in f:
            if l[0] == '|' or ':' not in l:
                continue
            cols = l.split(':')
            names.append(cols[0])
            if cols[1].startswith(' continuous.'):
                types.append(float)
            else:
                types.append(str)
        return names, types

## feat_names = list of names, feat_types = list of types
feat_names, feat_types = read_feature_descriptions('/home/mikael/Repos/Courses/DIT865/adult.names')

##for l in feat_names:
##    print l
##
##for l in feat_types:
##    print l

## A function that reads data
def read_data(filename, feat_names, feat_types):
    X = []
    Y = []
    with open(filename) as f:
        for l in f:
            cols = l.strip('\n.').split(', ')
            if len(cols) < len(feat_names): ## Skip empty lines/comments
                continue
            X.append( { n:t(c) for n, t, c in zip(feat_names, feat_types, cols) } )
            Y.append(cols[-1])
    return X, Y

## Read training and testing data

Xtrain, Ytrain = read_data('/home/mikael/Repos/Courses/DIT865/adult.data', feat_names, feat_types)











