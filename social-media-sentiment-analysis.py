import pandas
from bs4 import BeautifulSoup
import re
import nltk
import numpy
#only do next line once
#nltk.download() #download everything except panlex_lite
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from gensim.models import word2vec
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB
import sklearn.model_selection as cv
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from datetime import datetime

"""
author: Mahesh Gaya
project 5
description: Twitter Sentiment Analysis
"""
"""
Steps:
1. data clean up
	- some of the texts were combined when it should be. I arranged them 
		so that they are on different rows (excel)
	- Added "Result" and "Message" to know what columns we are using
"""

def clean_sentence( raw ):
	bs = BeautifulSoup(raw, "html.parser")
	letters_only = re.sub("[^a-zA-Z]"," ",bs.get_text())
	lower_case = letters_only.lower()
	words = lower_case.split()
	return words

def review_to_sentences( message_data, tokenizer):
	#didn’t seem to work without it, thanks StackOverflow
	#review = review.decode("utf-8")
	#strip out whitespace at beginning and end
	sentences_to_return = []
	for row in message_data:
		message = row.strip()
		raw_sentences = tokenizer.tokenize(message)
		sentences_list = []

		for sentence in raw_sentences:
			if len(sentence) > 0: #skip it if the sentence is empty
				cl_sent = clean_sentence(sentence)
				sentences_list += cl_sent
		sentences_to_return.append(sentences_list)
	return sentences_to_return

def make_attribute_vec(words, model, num_attributes):
	# Pre-initialize an empty numpy array (for speed)
	attribute_vec = numpy.zeros((num_attributes,),dtype="float32")
	nwords = 0.0
	# Loop over each word in the message and, if it is in the model’s
	# vocabulary, add its attribute vector to the total
	for word in words:
		if word in model.vocab:
			nwords = nwords + 1.0
			attribute_vec = numpy.add(attribute_vec,model[word])

	# Divide the result by the number of words to get the average
	attribute_vec = numpy.divide(attribute_vec,nwords)
	return attribute_vec

def calc_avg_accuarcy(list):
	list_sum = sum(list)
	return list_sum/len(list)

def main():
	tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
	test_data = pandas.read_csv("testdata.csv", delimiter=",", encoding = "ISO-8859-1")
	train_data = pandas.read_csv("training.csv", delimiter=",", encoding = "ISO-8859-1")
	test_sentences_for_all_messages = review_to_sentences(test_data["Message"],tokenizer) 
	#print(test_sentences_for_all_messages)
	train_sentences_for_all_messages = review_to_sentences(train_data["Message"],tokenizer) 
	#print(train_sentences_for_all_messages)
	num_attributes = 300 # Word vector dimensionality
	min_word_count = 40 # Minimum word frequency
	num_workers = 4 # Number of threads to run in parallel
	context = 10 # Context window size
	downsampling = 1e-3 # Downsample setting for frequent words
	# Initialize and train the model (this will take some time)

	test_model = word2vec.Word2Vec(test_sentences_for_all_messages, \
			workers=num_workers, size=num_attributes, \
			min_count = min_word_count, \
			window = context, sample = downsampling)
	#saves memory if you’re done training it
	test_model.init_sims(replace=True)
	#print(test_model.vocab)
	#print("christmas" in test_model.vocab)

	train_model = word2vec.Word2Vec(train_sentences_for_all_messages, \
			workers=num_workers, size=num_attributes, \
			min_count = min_word_count, \
			window = context, sample = downsampling)
	#saves memory if you’re done training it
	train_model.init_sims(replace=True)
	#print(train_model.vocab)
	#print("christmas" in train_model.vocab)

	#make the two lists into vector
	test_message_vector = []
	for i in range(len(test_sentences_for_all_messages)):
		vector = make_attribute_vec(test_sentences_for_all_messages[i], test_model, \
		num_attributes)
		test_message_vector.append(vector.tolist())
	
	#print("vector:", str(test_message_vector))

	train_message_vector = []
	for i in range(len(train_sentences_for_all_messages)):
		vector = make_attribute_vec(train_sentences_for_all_messages[i], test_model, \
		num_attributes)
		train_message_vector.append(vector.tolist())
	#print("vector:", str(train_message_vector))
	#print(len(train_message_vector))
	#print(len(train_data))

	train_message_vector = numpy.array(train_message_vector)
	
	#check_nan = numpy.isnan(train_message_vector)
	#if (True in check_nan): print("Yes")

	#convert any nan to zero
	train_message_vector = numpy.nan_to_num(train_message_vector)

	# Split the training data into testing and training data to check for accuracy
	def gaussianNB():
		print("\nGaussianNB")
		accuracy_list = []
		for i in range(0,20):
			(training_data, testing_data, training_target, testing_target) = \
				cv.train_test_split(train_message_vector, train_data["Result"],\
						test_size = 0.2) 
			gaussian_nb = GaussianNB()
			gaussian_nb.fit(training_data, training_target)
			prediction = gaussian_nb.predict(testing_data)
			accuracy = accuracy_score(prediction, testing_target)
			accuracy_list.append(accuracy)
			if (i < 10):
				print("0" + str(i) + " :--: " + str(accuracy))
			else:
				print(str(i) + " :--: " + str(accuracy))
		print(str("avg accuracy = ") + str(calc_avg_accuarcy(accuracy_list)))
	
	def bernoulliNB():
		print("\nBernoulliNB")
		accuracy_list = []
		for i in range(0,20):
			(training_data, testing_data, training_target, testing_target) = \
				cv.train_test_split(train_message_vector, train_data["Result"],\
						test_size = 0.2) 
			bernoulli_nb = BernoulliNB()
			bernoulli_nb.fit(training_data, training_target)
			prediction = bernoulli_nb.predict(testing_data)
			accuracy = accuracy_score(prediction, testing_target)
			accuracy_list.append(accuracy)
			if (i < 10):
				print("0" + str(i) + " :--: " + str(accuracy))
			else:
				print(str(i) + " :--: " + str(accuracy))
		print(str("avg accuracy = ") + str(calc_avg_accuarcy(accuracy_list)))

	def svm(kernel_type="rbf", n_degree=3):
		print("\nSVC: " + kernel_type + ", degree=" + str(n_degree))
		accuracy_list = []
		for i in range(0,20):
			(training_data, testing_data, training_target, testing_target) = \
				cv.train_test_split(train_message_vector, train_data["Result"],\
						test_size = 0.2) 
			svc = SVC(kernel=kernel_type, degree=n_degree)
			svc.fit(training_data, training_target)
			prediction = svc.predict(testing_data)
			accuracy = accuracy_score(prediction, testing_target)
			accuracy_list.append(accuracy)
			if (i < 10):
				print("0" + str(i) + " :--: " + str(accuracy))
			else:
				print(str(i) + " :--: " + str(accuracy))
		print(str("avg accuracy = ") + str(calc_avg_accuarcy(accuracy_list)))
	
	print(datetime.now().time())
	gaussianNB()
	print(datetime.now().time())
	bernoulliNB()
	print(datetime.now().time())
	svm("linear")
	print(datetime.now().time())
	svm(kernel_type="poly", n_degree=1)
	print(datetime.now().time())
	svm(kernel_type="poly")
	print(datetime.now().time())
	svm()
	print(datetime.now().time())
	svm(n_degree=1)
	print(datetime.now().time())

	
main()
