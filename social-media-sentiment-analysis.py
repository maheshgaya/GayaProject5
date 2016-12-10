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
description: Spam data analysis
data link: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
Number of instances: 5574
"""


def print_pretty(msg, datum):
	separation = "-"
	for i in range(len(msg)):
		separation = str(separation + "-")
	print(separation)
	print(str(msg) + ":")
	print(str(datum))
	print(separation)

def main():
	data = pandas.read_csv("SMSSpamCollection.csv", delimiter=",")
	print_pretty("Data Shape", data.shape)
	messages = data['Messages'][0:]
	print_pretty("Messages", messages)
main()
