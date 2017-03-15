Requirements: Java - JDK, Python modules - scipy, nltk, re, numpy, practnlptools, sklearn

The path for "stanford-ner.jar" which is in "stanford-ner-2014-06-16" has to be added to CLASSPATH. 

The code is written in Python. I have used Python - version 2.7.13

First Run the "Extracting_features.py" file
	- While submitting this code, code lines for computing NER and Chunks have been commented as the code takes long time (aaproximately 5 hours) to extract the features.
	- The above computed files are already present in the features folder.
	- In case to run, please uncomment those lines which are at the end of the code.
	
The train data and test data are in Data folder. 
    - Dummy class "q" has been assigned to every question in the test_data.
Next run the "Training and Testing.py" file.

The output file classes will be in the Data folder.