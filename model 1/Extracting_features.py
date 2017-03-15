import re
import nltk
from nltk.tag.stanford import StanfordNERTagger
from practnlptools.tools import Annotator
from readproperties import read_property



def preprocessing(raw_sentence):
    sentence= re.sub(r'[$|.|!|"|(|)|,|;|`|\']',r'',raw_sentence)
    return sentence

def file_preprocess(filename):
	questions=[]
	f=open(filename,'r')
	fi=open(read_property('word_features_train_path'),"w")
	lines=f.readlines()
	for line in lines:
		line=line.rstrip('\n')
		line=preprocessing(line)
		sentence=""
		words=line.split()
		for i in range(0,len(words)):
			if not(i==0):
				sentence=sentence+(words[i])+" "
		fi.write(sentence+"\n")
		questions.append(sentence)
	f.close()
	fi.close()
	return questions



def POS_Tags(questions):
      fi=open(read_property('POS_features_train_path'),"w")
      for sentence in questions:
            text = nltk.word_tokenize(sentence)
            pos_seq=nltk.pos_tag(text)           
            pos_tags=""
            for pos in pos_seq:
                  pos_tags=pos_tags+pos[1]+" "
	    fi.write(pos_tags+"\n")
      fi.close()
     
      
    
def NER(questions):
      fi=open(read_property('NER_features_train_path'),"w")
      st = StanfordNERTagger(read_property('StanfordNerClassifier'),read_property('StanfordNerJarPath'))
      for sentence in questions:
            ner=st.tag(sentence.split())
            ner_tag=""
            for n in ner:
                  ner_tag=ner_tag+n[1]+" "
	    fi.write(ner_tag+"\n")
      fi.close()



    
def Chunks(questions):
      fi=open(read_property('Chunk_features_train_path'),"w")
      annotator=Annotator()
      for sentence in questions:
	    chunks=annotator.getAnnotations(sentence)['chunk']
            chunk=""
            for elem in chunks:
                  chunk=chunk+elem[1]+" "
	    fi.write(chunk+"\n")
      fi.close()
      

def Wordshapes(questions):
    fi = open(read_property('wordshapes_train_path'),"w")
    for i in range(len(questions)):
       line = questions[i]
       words = nltk.word_tokenize(line)     
       mixed = 0
       all_up = 0
       all_low = 0
       numeric = 0
       for word in words:
           if word.islower() == True:
               all_low = all_low + 1
           elif word.isupper() == True:
               all_up = all_up + 1
           elif word.isupper() == False and word.islower() == False and word.isdigit() == False:
               mixed = mixed + 1
           elif word.isdigit() == True:
               numeric = numeric + 1
       sent = str(all_up) + ' ' + str(mixed) + ' ' + str(all_low) + ' ' + str(numeric)
       fi.write(sent+'\n')
    fi.close()  
                  
def Wordshapes_test(questions):
    fi = open(read_property('wordshapes_test_path'),"w") 
    for i in range(len(questions)):
       line = questions[i]
       words = nltk.word_tokenize(line)     
       mixed = 0
       all_up = 0
       all_low = 0
       numeric = 0
       for word in words:
           if word.islower() == True:
               all_low = all_low + 1
           elif word.isupper() == True:
               all_up = all_up + 1
           elif word.isupper() == False and word.islower() == False and word.isdigit() == False:
               mixed = mixed + 1
           elif word.isdigit() == True:
               numeric = numeric + 1
       sent = str(all_up) + ' ' + str(mixed) + ' ' + str(all_low) + ' ' + str(numeric)
       fi.write(sent+'\n')
    fi.close()  



def file_preprocess_test(filename):
	questions=[]
	f=open(filename,'r')
	fi=open(read_property('word_features_test_path'),"w")
	lines=f.readlines()
	for line in lines:
		line=line.rstrip('\n')
		line=preprocessing(line)
		sentence=""
		words=line.split()
		for i in range(0,len(words)):
			if not(i==0):
				sentence=sentence+(words[i])+" "
		fi.write(sentence+"\n")
		questions.append(sentence)
	f.close()
	fi.close()
	return questions
 

    
def POS_Tags_test(questions):
      fi=open(read_property('POS_features_test_path'),"w")
      for sentence in questions:
            text = nltk.word_tokenize(sentence)
            pos_seq=nltk.pos_tag(text)
            pos_tags=""
            for pos in pos_seq:
                  pos_tags=pos_tags+pos[1]+" "
	    fi.write(pos_tags+"\n")
      fi.close()
      
      
      
  
def NER_test(questions):
      fi=open(read_property('NER_features_test_path'),"w")
      st = StanfordNERTagger(read_property('StanfordNerClassifier'),read_property('StanfordNerJarPath'))
      for sentence in questions:
            ner=st.tag(sentence.split())
            ner_tag=""
            for n in ner:
                  ner_tag=ner_tag+n[1]+" "
	    fi.write(ner_tag+"\n")
      fi.close()
      


    
def Chunks_test(questions):
      fi=open(read_property('Chunk_features_test_path'),"w")
      annotator=Annotator()
      for sentence in questions:
	    chunks=annotator.getAnnotations(sentence)['chunk']
            chunk=""
            for elem in chunks:
                  chunk=chunk+elem[1]+" "
	    fi.write(chunk+"\n")
            
      fi.close()
      


filename_train=read_property('trainingfilepath')
questions=file_preprocess(filename_train)
#Wordshapes(questions)
#POS_Tags(questions)
#NER(questions)
#Chunks(questions)


filename_test = read_property('testfilepath')
questions_test = file_preprocess_test(filename_test)
#Wordshapes_test(questions_test)
#POS_Tags_test(questions_test)
#NER_test(questions_test)
#Chunks_test(questions_test)
