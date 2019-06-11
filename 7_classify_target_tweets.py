# -*- coding: utf-8 -*-

"""
	Created by:	Shaheen Syed
	Date: 		July 2018

	Use the created machine learning classifier to infer the sentiment class of the target tweets.

"""

# packages and modules
import numpy as np
from helper_functions import *
from database import MongoDatabase
from sklearn.externals import joblib

"""
	Script starts here
"""

if __name__ == "__main__":

	# create logging to console
	set_logger()

	# verbose
	logging.info('Start: {} '.format(__file__))

	# create database connection
	db = MongoDatabase()

	# load classifier
	clf = joblib.load(os.path.join('files', 'ml_models', 'LinearSVC.pkl'))

	# read labels for target tweets that have been manually labeled and convert to dictionary with key = tweet ID and value = label
	true_labels = {d['tweet_id'] : d['label'] for d in db.read_collection(collection = 'manual_tweets_raw') }

	# load tweets for which we want to infer the sentiment label
	D = db.read_collection(collection = 'target_tweets')

	# create empty numpy array so we can retrieve labels later on somewhat faster
	labels = np.zeros((D.count(), 3), dtype = np.int)

	# loop over each target tweet
	for i, d in enumerate(D):

		logging.debug('	- Processing tweet {}/{}'.format(i + 1, D.count()))

		# check if we have a true label for the target tweet, if so, skip prediction and use true label
		if d['tweet_id'] in true_labels:
			# the labels are stored fully written, for example, positive, and we need to go to the coded version, that is 2 for positive
			d['label'] = get_sentiment_code(true_labels[d['tweet_id']])
		else:
			# infer sentiment label from classifier
			d['label'] = int(clf.predict([d['text']])[0][0])


		# update the document in the database
		db.update_collection(collection = 'target_tweets', doc = d)

		# add to label array
		labels[i] = (d['tweet_id'], d['label'], get_tweet_type_code(d['tweet_type']))

		# increment counter
		# i +=1

	# location for the labels array
	labels_location = os.path.join('files', 'labels')
	# make sure directory exists
	create_directory(labels_location)
	# save labels array
	joblib.dump(labels, os.path.join(labels_location, 'labels.pkl'))
