import pandas
from bs4 import BeautifulSoup
import re
import nltk
#only do next line once
#nltk.download() #download everything except panlex_lite
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

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

def print_pretty(msg, datum):
	separation = "-"
	for i in range(len(msg)):
		separation = str(separation + "-")
	print(separation)
	print(str(msg) + ":")
	print(str(datum))
	print(separation)

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
	# Loop over each word in the review and, if it is in the model’s
	# vocaublary, add its attribute vector to the total
	for word in words:
		if word in model.vocab:
			nwords = nwords + 1.0
			attribute_vec = numpy.add(attribute_vec,model[word])

	# Divide the result by the number of words to get the average
	attribute_vec = numpy.divide(attribute_vec,nwords)
	return attribute_vec


def main():
	tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
	test_data = pandas.read_csv("testdata.csv", delimiter=",", encoding = "ISO-8859-1")
	train_data = pandas.read_csv("training.csv", delimiter=",", encoding = "ISO-8859-1")
	test_sentences_for_all_messages = review_to_sentences(test_data["Message"],tokenizer) 
	#print(test_sentences_for_all_messages)
	train_sentences_for_all_messages = review_to_sentences(train_data["Message"],tokenizer) 
	#print(train_sentences_for_all_messages)

	
main()
