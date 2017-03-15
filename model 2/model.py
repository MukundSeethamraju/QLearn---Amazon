from os import path,listdir
import sys
import re
import fileinput
from pprint import pprint
from time import time
import logging as log
import numpy
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import features
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
import configuration
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.metrics import accuracy_score
from sklearn import cross_validation




class Classifier:

    def __init__(self, init_data=None):
        self.data = init_data
        self.model = self.build_model()
        

    def build_model(self):
        model = Pipeline([
            ('union', FeatureUnion([
                ('words', TfidfVectorizer(max_df=0.25, ngram_range=(1, 4),
                                          sublinear_tf=True, max_features=5000)),
                                          
                ('relword',features.RelatedWordVectorizer(max_df=0.75, ngram_range=(1, 4),
                                                  sublinear_tf=True)),
                
                ('pos', features.TagVectorizer(max_df=0.75, ngram_range=(1, 4),
                                       sublinear_tf=True)),
                 
            ])),
            
            ('clf', LinearSVC()),
           
        ])
       
        
        return model

    def train_model(self):
        log.debug("Training model...")
        self.model.fit(self.data.data, self.data.target)

    
    
    def save_model(self,filename):
        log.debug("Saving model to file: " + filename)      
        joblib.dump(self.model,path.join(configuration.MODEL_DIR,filename))
        
    def load_model(self,pklfile):
        log.debug("Loading model from file: " + pklfile)
        self.model = joblib.load(pklfile)
        return self
    
    def predict(self, doc):
        
        clas = self.model.predict([doc])
        
        return clas
        
'''
class Classifier:

    def __init__(self, init_data=None):
        self.data = init_data
        self.model = self.build_model()
        

    def build_model(self):     
        model = Pipeline([
            ('union', FeatureUnion([
                ('words', TfidfVectorizer(max_df=0.25, ngram_range=(1, 4),
                                          sublinear_tf=True, max_features=5000)),
                                          
                ('relword',features.RelatedWordVectorizer(max_df=0.75, ngram_range=(1, 4),
                                                  sublinear_tf=True)),
                
                ('pos', features.TagVectorizer(max_df=0.75, ngram_range=(1, 4),
                                       sublinear_tf=True)),
                 
            ])),
            
            ('clf', LinearSVC()),
           
        ])
       
        
        return model

    def train_model(self):
        
        log.debug("Training model...")
        self.model.fit(self.data.data, self.data.target)

    
    
    def save_model(self,filename):
        if filename=="":
            log.ERROR("Empty file name")
        log.debug("Saving model to file: " + filename)      
        joblib.dump(self.model,path.join(configuration.MODEL_DIR,filename))
        
    def load_model(self,pklfile):
        if pklfile=="":
            log.ERROR("Empty file name")
        log.debug("Loading model from file: " + pklfile)
        self.model = joblib.load(pklfile)
        return self
    
    def predict(self, doc):
        clas = self.model.predict([doc])
        
        return clas
'''  
def classify_question_type(text):
    clf = Classifier()
    clf.load_model(path.join(configuration.MODEL_DIR,"Train.pkl"))
    clas = clf.predict(text) 
    clas = numpy.array(clas).tolist()
    print clas
    return clas


def load_data(filenames):

    data = [] 
    target = []
    fine_target = [] 
    
    data_re = re.compile(r'(\w+):(\w+) (.+)')     

    for line in fileinput.input(filenames):
        d = data_re.match(line)        
        if not d:
            raise Exception("Invalid format in file {} at line {}"
                            .format(fileinput.filename(), fileinput.filelineno()))
        
        target.append(d.group(1))
       
        fine_target.append(d.group(2))
        
        data.append(d.group(3))            

    return Bunch(
        data=numpy.array(data),
        target=numpy.array(target),
        target_names=set(target), )
    
if __name__ == "__main__":

    data = load_data("./data/Train_data.txt")
    print data 
    clf = Classifier(data)
    
    
    clf.train_model()
    
    clf.save_model("Train.pkl")
    
    clf.load_model(path.join(configuration.MODEL_DIR,"Train.pkl"))
    clf=joblib.load(path.join(configuration.MODEL_DIR,"Train.pkl"))
    
    
    f = open('./data/test.txt', 'r')
    lines = []
    for line in f:
        lines.append(line)
    f.close()
    
    fi = open('./data/classes_output.txt','w')
    for line in lines:
        fi.write(str(classify_question_type(line)[0]) + '\n')
    fi.close()