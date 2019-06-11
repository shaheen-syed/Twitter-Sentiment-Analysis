# coding: utf-8

"""
	Created by Shaheen Syed
	Date: August 2018
"""

import logging # logging to console and file
import os
import glob2 # read directories
import sys
import spacy # for NLP
import re # to use regular expressions
import string # to get a list of punctation
import csv # to read and write CSV files
import pickle # to save/read objects
from datetime import datetime


def set_logger(folder_name = 'logs'):

	"""
		Set up the logging to console layout

		Parameters
		----------
		folder_name : string, optional
				name of the folder where the logs can be saved to

	"""

	# create the logging folder if not exists
	create_directory(folder_name)

	# define the name of the log file
	log_file_name = os.path.join(folder_name, '{:%Y%m%d%H%M%S}.log'.format(datetime.now()))

	# set up the logger layout to console
	logging.basicConfig(filename=log_file_name, level=logging.NOTSET)
	console = logging.StreamHandler()
	formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)
	logger = logging.getLogger(__name__)


def create_directory(name):

	"""
		Create directory if not exists

		Parameters
		----------
		name : string
				name of the folder to be created

	"""

	try:
		if not os.path.exists(name):
			os.makedirs(name)
			logging.info('Created directory: {}'.format(name))
	except Exception, e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def read_directory(directory):

	"""

		Read file names from directory recursively

		Parameters
		----------
		directory : string
					directory/folder name where to read the file names from

		Returns
		---------
		files : list of strings
    			list of file names
	"""
	
	try:
		return glob2.glob(os.path.join( directory, '**' , '*.*'))
	except Exception, e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def read_plain_text(file_name, read_lines = False):

	"""
		Save string as text file

		Parameters
		----------
    	file_name : string
    			The name of the file you want to give it
    	read_lines : Boolean (optional)
    		Return the content of the file as a list by splitting on new lines

    	Returns
    	--------
    	plain_text : string
    		the plain text from the .txt file

	"""

	try:

		# save data to folder with name
		with open(file_name, 'rb') as f:

			if not read_lines:
				# read the content and return
				return f.read()
			else:
				# read content and return as list by splitting on lines
				return f.readlines()

	except Exception, e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)

def read_csv(filename, folder = None):

	"""
		Read CSV file and return as a list

		Parameters
		---------
		filename : string
			name of the csv file
		folder : string (optional)
			name of the folder where the csv file can be read

		Returns
		--------
		csv : list
			content of the csv file as a list

	"""

	if folder is not None:
		filename = os.path.join(folder, filename)
	
	try:
		# increate CSV max size
		csv.field_size_limit(sys.maxsize)
		
		# open the filename
		with open(filename, 'rb') as f:
			# create the reader
			reader = csv.reader(f)
			# return csv as list
			return list(reader)
	except Exception, e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)

def save_dic_to_csv(dic, file_name, folder):

	"""
		Save a dictionary as CSV (comma separated values)

		Parameters
		----------
		dic : dic
    			dictionary with key value pairs
    	name : string
    			The name of the file you want to give it
    	folder: string
    			The folder location
	"""
	
	try:

		# create folder name as directory if not exists
		create_directory(folder)

		# check if .csv is used as an extension, this is not required
		if file_name[-4:] == '.csv':
			file_name = file_name[:-4]

		# create the file name
		file_name = os.path.join(folder, file_name + '.csv')

		# save data to folder with name
		with open(file_name, "w") as f:

			writer = csv.writer(f, lineterminator='\n')
			
			for k, v in dic.items():
				writer.writerow([k, v])

	except Exception, e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)

def save_pickle(obj, file_name, folder):
	
	"""
		Save python object with pickle

		Parameters
		----------
		obj : object
			object that need to be pickled
		name : string
			name of the file
		folder : string
			location of folder to store pickle file in
	"""

	# create folder if not exists
	create_directory(folder)

	# check if .pkl is used as an extension, this is not required
	if file_name[-4:] == '.pkl':
		file_name = file_name[:-4]

	# save as pickle
	with open(os.path.join(folder, file_name + '.pkl'), 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(file_name, folder = None):

	"""
		Load python object with pickle

		Parameters
		---------
		file_name : string
			file name of the pickle file to load/open
		folder : string (optional)
			name of folder if not already part of the file name

		Returns
		-------
		pickle file
	"""

	# check if .pkl is used as an extension, this is not required
	if file_name[-4:] == '.pkl':
		file_name = file_name[:-4]

	# check if folder has been sent
	if folder is not None:
		file_name = os.path.join(folder, file_name)

	# open file with pickle and return
	with open(os.path.join(file_name + '.pkl'), 'rb') as f:
		return pickle.load(f)

def clean_tweet(text):

	"""
		Preprocessing tweet text

		Parameters
		---------
		text : string
			content of a tweet

		Returns
		--------
		text: string
			preprocessed tweet
	"""

	# replace new lines
	text = re.sub(r'\n', ' ', text)
	# replace ampersand character
	text = text.replace('&amp;', ' and ')
	# replace @
	text = re.sub(r'@.*?( |$)', 'USERTAG ', text)
	# replace URL
	text = re.sub(r'http[s]{0,1}.*?( |$)', 'URLTAG ', text)
	# replace hashtag
	text = text.replace('#', '')
	# replace contractions
	text = decontract_text(text)
	# replace emoticons
	text = replace_emoticons(text)
	# replace emojis
	text = replace_emojis(text)
	# replace repeating charachtes : happyyyyy -> happyy
	text = replace_repeating_characters(text)
	# replace punctuation
	text = replace_punctuation(text)
	# replace specific characters
	text = replace_specific_characters(text)
	# replace double spaced
	text = text.replace("  ", " ").replace("  ", " ")
	# trim leading and trailing spaces
	text = text.strip()

	return text

def decontract_text(text):

	"""
		Decontract text, for example don't -> do not

		Parameters
		---------
		text : string
			content of a tweet

		Returns
		--------
		text: string
			decontracted text of the tweet
	"""

	# specific
	text = re.sub(r"won't", "will not", text)
	text = re.sub(r"can\'t", "can not", text)

	# general
	text = re.sub(r"n\'t", " not", text)
	text = re.sub(ur"n\u2019t", " not", text)
	text = re.sub(r"\'re", " are", text)
	text = re.sub(r"\'s", " is", text)
	text = re.sub(ur"\u2019s"," is", text)
	text = re.sub(r"\'d", " would", text)
	text = re.sub(r"\'ll", " will", text)
	text = re.sub(r"\'t", " not", text)
	text = re.sub(r"\'ve", " have", text)
	text = re.sub(r"\'m", " am", text)

	return text


def replace_emoticons(text, placeholder_pos = ' HAPPYEMOTICON ', placeholder_neg = ' SADEMOTICON '):

	"""
		Replace emoticons to a placeholder

		Parameters
		----------
		text: string
			text content of the tweet
		placeholder_pos : string (optional)
			placeholder for positive emoticons
		placeholder_neg : string (optional)
			placeholder for negative emoticons

		Returns
		--------
		text : string
			text of tweet with replaced emoticons

	"""

	emoticons_pos = [":)", ":-)", ":p", ":-p", ":P", ":-P", ":D",":-D", ":]", ":-]", ";)", ";-)", ";p", ";-p", ";P", ";-P", ";D", ";-D", ";]", ";-]", "=)", "=-)", "<3"]
	emoticons_neg = [":o", ":-o", ":O", ":-O", ":(", ":-(", ":c", ":-c", ":C", ":-C", ":[", ":-[", ":/", ":-/", ":\\", ":-\\", ":n", ":-n", ":u", ":-u", "=(", "=-(", ":$", ":-$"]

	# replace positive emoticons by placeholder
	for e in emoticons_pos:
		text = text.replace(e, placeholder_pos)

	# replace negative emoticons by placeholder
	for e in emoticons_neg:
		text = text.replace(e, placeholder_neg)

	return text

def replace_emojis(text, placeholder_pos = ' HAPPYEMOJI ', placeholder_neg = ' SADEMOJI '):

	"""
		Replace emojis to a placeholder

		Parameters
		----------
		text: string
			text content of the tweet
		placeholder_pos : string (optional)
			placeholder for positive emoji
		placeholder_neg : string (optional)
			placeholder for negative emoji

		Returns
		--------
		text : string
			text of tweet with replaced emojis

	"""

	# define positive emojis
	emoji_pos = [u'\U0001f600',u'\U0001f601',u'\U0001f602',u'\U0001f923',u'\U0001f603',u'\U0001f604',u'\U0001f605',u'\U0001f606',
				u'\U0001f609',u'\U0001f60a',u'\U0001f60b',u'\U0001f60e',u'\U0001f60d',u'\U0001f618',u'\U0001f617',u'\U0001f619',
				u'\U0001f61a',u'\\U000263a',u'\U0001f642',u'\U0001f917']

	# define negative emojis
	emoji_neg = [u'\\U0002639',u'\U0001f641',u'\U0001f616',u'\U0001f61e',u'\U0001f61f',u'\U0001f624',u'\U0001f622',u'\U0001f62d',
				u'\U0001f626',u'\U0001f627',u'\U0001f628',u'\U0001f629',u'\U0001f62c',u'\U0001f630',u'\U0001f631',u'\U0001f633',
				u'\U0001f635',u'\U0001f621',u'\U0001f620',u'\U0001f612']

	# replace positive emojis by placeholder
	for e in emoji_pos:
		text = text.replace(e, placeholder_pos)

	# replace negative emojis by placeholder
	for e in emoji_neg:
		text = text.replace(e, placeholder_neg)

	return text

def replace_repeating_characters(text):

	"""
		Replace repeating characters, for example, loveeeee into lovee
			- more than 2 of the same

		Parameters
		----------
		text: string
			text content of the tweet

		Returns
		--------
		text : string
			text of the tweet with repeating characters shortened

	"""

	return re.sub(r'(.)\1{2,}', r'\1\1', text)

def replace_punctuation(text):

	"""
		Replace all punctuation with space
		Note that this should be done after emoticons are replaced by a placeholder

		Parameters
		----------
		text : string
			text content of the tweet

		Returns
		--------
		text : string
			text of the tweet with punctuation replaced by a space
	"""
	
	for c in string.punctuation:
		text = text.replace(c," ")

	return text

def replace_specific_characters(text):

	"""
		Replace unwanted characters

		Parameters
		----------
		text : string
			text content of the tweet

		Returns
		--------
		text : string
			text of the tweet with some characters replaced by a space
	"""

	text = text.replace(u'\u201c', ' ')	# double opening quotes
	text = text.replace(u'\u201d', ' ')	# double closing quotes
	text = text.replace(u'\u2014', ' ')	# -
	text = text.replace(u'\u2013', ' ') # -
	text = text.replace(u'\u2026', ' ') # horizontal elipsses ...

	return text

def get_tokens(text):

	"""
		Get individual tokens from text

		Parameters
		---------
		text : string
			text content

		Returns
		--------
		tokens : list
			list of tokens

	"""

	# split on whitespace
	tokens = text.split(" ")

	# remove space tokens and numbers, and convert to lowercase
	tokens = [x.lower() for x in tokens if x != " " and not x.isdigit()]

	return tokens


def get_lemma(spacy_doc):

	"""
		get the lemma of the tokens, for example, policies -> policy

		Parameters
		----------
		spacy_doc : spacy nlp object

		Returns
		---------
		tokens: tokens with lemma

	"""


	try:
		return  [token.lemma_ for token in spacy_doc]
	except Exception, e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def setup_spacy():

	# setting up spacy and loading an English corpus
	nlp = spacy.load('en')

	# load the same corpus but in a different way (depends if there is a symbolic link)
	#nlp = spacy.load('en_core_web_sm')

	return nlp

def word_tokenizer(text):

	"""
		Function to return individual words from text. Note that lemma of word is returned excluding numbers, stopwords and single character words

		Parameters
		----------
		text : spacy object
			plain text wrapped into a spacy nlp object

		Returns
	"""

	# start tokenizing
	try:
		# Lemmatize tokens, remove punctuation, remove single character tokens and remove stopwords.
		return  [token.lemma_ for token in text if token.is_alpha and not token.is_stop and len(token) > 1]
	except Exception, e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)

def get_sentiment_code(label):

	"""
		Get the numeric code for the sentiment label
		
		Parameters
		----------
		label : string
			written sentiment label, for example, positive, negative, or neutral

		Returns
		--------
		code: int		
			negative = 0
			neutral = 1
			positive = 2
	
	"""

	# negative or '0' gets encoded into 0
	if label.lower() == 'negative' or label == '0':
		return 0

	# neutral or '2' gets encoded into 1
	elif label.lower() == 'neutral' or label == '2':
		return 1
			
	# positive or '4' gets encoded into 2
	if label.lower() == 'positive' or label == '4':
		return 2
	
	# unknown label
	else:
		logging.error('Unhandled sentiment label: {}'.format(label))
		exit(1)


def get_sentiment_label(code):
	
	"""
		Get the sentiment label from the sentiment code. For example, return 'positive' for the sentiment code 2

		Parameters
		----------
		code: int
			code for sentiment class, either 0, 1 or 2

		Returns
		-------
		label : string
			returns the sentiment class label fully written, for instance 'positive'

	"""

	try:

		sentiment = {}

		sentiment[0] = 'negative'
		sentiment[1] = 'neutral'
		sentiment[2] = 'positive'

		return sentiment[code]
	except Exception, e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def get_tweet_type_code(tweet_type):

	"""
		Get code for tweet type.
		- for example, code the type 'interdisciplinary' as 0, etc.

		Parameters
		----------
		tweet_type : string
			interdisciplinary or transdisciplinary or multidisciplinary

		Returns
		code : int
			coded version of the tweet type, either 0, 1 or 2
	"""

	try:	
		tweet_types = {}
		tweet_types['interdisciplinary'] = 0
		tweet_types['transdisciplinary'] = 1
		tweet_types['multidisciplinary'] = 2

		return tweet_types[tweet_type]

	except Exception, e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)

def get_tweet_type_from_code(code):

	"""
		Get the fully written tweet type from encoding
		- for example, return interdisciplinary when code 0 is send as argument

		Parameters
		---------
		code : int
			0 = interdisciplinary, 1 = transdisciplinary, 2 = multidisciplinary

		Returns
		---------
		tweet_type : string
			interdisciplinary or transdisciplinary or multidisciplinary
	"""

	tweet_types = ['interdisciplinary','transdisciplinary','multidisciplinary']

	return tweet_types[code]
