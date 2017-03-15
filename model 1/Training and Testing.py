import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer, TfidfVectorizer
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.svm import LinearSVC
from readproperties import read_property


train_class=[]
f=open(read_property('trainingfilepath'),'r')
lines=f.readlines()
for line in lines:
	line=line.rstrip('\n')
    	if not (line=="\n"):
        	train_class.append((line.split()[0]))



print "Extracting word features for training"
f=open(read_property('word_features_train_path'),"r")
questions=[]
for lines in f:
	l=lines.split()
	words=""
	for w in l:
		words=words+w+" "
	questions.append(words)		
vectorizer_words= CountVectorizer(lowercase = True,ngram_range = (1,4),stop_words= ["a","all","also","and","many","last",'not','inc','ie','.', ',', '"', "'", "?", "!", ':', ';', '(', ')', '[', ']', '{', '}'])
 #["a", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","among", "amongst", "amoungst", "and", "any", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "beforehand", "being", "beside", "besides", "both", "bottom","but", "by", "cannot", "cant", "co", "con", "could", "couldnt", "de", "describe", "detail", "do", "done", "down", "eg", "either","else", "enough", "etc", "even", "ever", "every", "except", "few", "find", "for", "from", "front", "full", "further", "get", "give", "had", "has", "hasnt", "have", "hence", "hereafter", "hereby", "herein", "hereupon", "however", "ie", "if", "inc", "indeed", "interest", "into", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "ltd", "made", "many", "meanwhile", "might", "mill", "more", "moreover", "most", "mostly", "move", "much", "must", "namely", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing","of", "off", "often", "on", "or", "otherwise","out", "over", "own","part", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "should", "show", "side", "since", "sincere", "so", "some", "somehow", "still", "such", "than", "that", "the", "their", "then", "thence", "thereby", "therefore", "thereupon", "thickv", "thin", "this", "though", "through", "throughout", "thru", "thus", "to", "too", "un", "under", "upon", "very", "via", "well", "whither", "will", "with", "within", "yet",
                            
features_words = vectorizer_words.fit_transform(questions)
f.close()





print "Extracting POS features for training"
f=open(read_property('POS_features_train_path'),"r")
questions=[]
for lines in f:
	l=lines.split()
	words=""
	for w in l:
		words=words+w+" "
	questions.append(words)		
vectorizer_POS= CountVectorizer(ngram_range=(1,4))
features_POS = vectorizer_POS.fit_transform((questions))  
f.close()




print "Extracting NER features for training"
f=open(read_property('NER_features_train_path'),"r")
questions=[]
for lines in f:
	l=lines.split()
	words=""
	for w in l:
		words=words+w+" "
	questions.append(words)		
vectorizer_NER= CountVectorizer()
features_NER = vectorizer_NER.fit_transform((questions))
f.close()




print "Extracting Chunk features for training"
f=open(read_property('Chunk_features_train_path'),"r")
questions=[]
for lines in f:
	l=lines.split()
	words=""
	for w in l:
		words=words+w+" "
	questions.append(words)		
vectorizer_Chunk= CountVectorizer() #min_df = 1
features_Chunk = vectorizer_Chunk.fit_transform((questions))  
f.close()


print "Extracting wordshapes features for training"
f=open(read_property('wordshapes_train_path'),"r")
questions=[]
for lines in f:
	l=lines.split()
	words=""
	for w in l:
		words=words+w+" "
	questions.append(words)		
vectorizer_wordshapes = CountVectorizer()
features_wordshapes = vectorizer_wordshapes.fit_transform((questions))  
f.close()


print "Extracting word features for testing"
f=open(read_property('word_features_test_path'),"r")
questions=[]
for lines in f:
	l=lines.split()
	words=""
	for w in l:
		words=words+w+" "
	questions.append(words)		
features_words_n = vectorizer_words.transform(questions)
f.close()

print "Extracting POS features for testing"
f=open(read_property('POS_features_test_path'),"r")
questions=[]
for lines in f:
	l=lines.split()
	words=""
	for w in l:
		words=words+w+" "
	questions.append(words)		

features_POS_n = vectorizer_POS.transform((questions))  
f.close()



print "Extracting NER features for testing"
f=open(read_property('NER_features_test_path'),"r")
questions=[]
for lines in f:
	l=lines.split()
	words=""
	for w in l:
		words=words+w+" "
	questions.append(words)		

features_NER_n = vectorizer_NER.transform((questions))  
f.close()




print "Extracting Chunk features for testing"
f=open(read_property('Chunk_features_test_path'),"r")
questions=[]
for lines in f:
	l=lines.split()
	words=""
	for w in l:
		words=words+w+" "
	questions.append(words)		

features_Chunk_n = vectorizer_Chunk.transform((questions))  
f.close()


print "Extracting wordshapes features for testing"
f=open(read_property('wordshapes_test_path'),"r")
questions=[]
for lines in f:
	l=lines.split()
	words=""
	for w in l:
		words=words+w+" "
	questions.append(words)		

features_wordshapes_n = vectorizer_wordshapes.transform((questions))  
f.close()






features=hstack((features_words,features_POS),format='csr')
features_train = hstack((features,features_NER),format='csr')
features_Train=hstack((features_train,features_Chunk),format='csr')
features_Train = hstack((features_Train,features_wordshapes),format = 'csr')


features_n=hstack((features_words_n,features_POS_n),format='csr')
features_test=hstack((features_n,features_NER_n),format='csr')
features_test=hstack((features_test,features_Chunk_n),format='csr')
features_test=hstack((features_test,features_wordshapes_n),format='csr')


self = LinearSVC(loss='l2', dual=False, tol=1e-3,multi_class = 'crammer_singer')
self = LinearSVC.fit(self, features_Train, train_class)
test_class = LinearSVC.predict(self, features_test)



hits=0.00
fi=open(read_property('output_path'),"w")
for i in range(0,len(test_class)):
    print test_class[i]," : ",questions_test[i],"\n"
    str_l=test_class[i]," : ",questions_test[i],"\n"
    fi.write(test_class[i]+" : ")
    fi.write(questions_test[i]+"\n")
fi.close()

'''
for i in range(0,len(test_class)):
	if test_class[i]== train_class[i]:
		hits=hits+1
print "Number of hits = ",hits
print "The accuracy is ",((hits/len(test_class))*100.0)," %"
'''