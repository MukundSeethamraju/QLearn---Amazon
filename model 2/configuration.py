# Configu ration for nltk Stanford interface (doesn't work very well)
import os
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib

NER_DIR = os.path.join(os.path.dirname(__file__), "stanford-ner")
NER_JAR = os.path.join(NER_DIR, "stanford-ner.jar")
NER_MODEL = "english.muc.7class.distsim.crf.ser.gz"
NER_MODEL = "english.conll.4class.distsim.crf.ser.gz"
NER_MODEL_PATH = os.path.join(NER_DIR, "classifiers", NER_MODEL)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
