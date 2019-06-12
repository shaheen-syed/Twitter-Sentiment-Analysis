# -*- coding: utf-8 -*-

"""
	Created by:	Shaheen Syed
	Date: 		July 2018

	Step 2 – Parse target tweets
	----------------------------
	This script reads all the .txt files created while running the script in step 1, and parses out the individual tweets and relevant fields. It then saves each tweet as a 
	document in a MongoDB database. The script knows if tweets have already been inserted previously, so there is no need to check for this.

	How to run:
	python 2_parse_target_tweets.py
	
"""

# packages and modules
import json	# to work convert plain text to json
import re # for regular expression
from helper_functions import *
from database import MongoDatabase


"""
	Script starts here
"""
if __name__ == "__main__":

	# create logging to console
	set_logger()

	logging.info('Start: {} '.format(__file__))

	# create database connection
	db = MongoDatabase()

	# location of target tweets
	location_tweets = os.path.join('files', 'target_tweets')

	# modes of research
	modes_of_research = ['interdisciplinary', 'multidisciplinary','transdisciplinary']

	# process tweets for each mode of research
	for mode in modes_of_research:

		logging.info('Processing mode of research: {}'.format(mode))

		# read tweets files
		F = read_directory(os.path.join(location_tweets, mode))

		# tracker to keep track of processed tweet ids
		tweet_tracker = set(['{}{}'.format(x['tweet_type'], x['id']) for x in db.read_collection(collection = 'raw_tweets')])

		# loop over each file, read content, parse relevant fields, save to db
		for i, f in enumerate(F):

			logging.info('Processing file {} {}/{}'.format(f, i + 1, len(F)))

			# read tweets from file (as list)
			tweets = read_plain_text(f, read_lines = True)

			# loop over each tweet
			for tweet in tweets:

				# convert string to json
				tweet = json.loads(tweet)

				# check if tweet ID already processed
				if '{}{}'.format(mode, str(tweet['id'])) not in tweet_tracker:

					# add to tracker so we won't process the same id + type again
					tweet_tracker.add(mode + str(tweet['id']))

					# create new document so we can save it to the database
					doc = {}
					# save the tweet ID (this is the unique identifier for each tweet)
					doc['id'] = tweet['id']
					# we refer to mode of research as tweet_type, so this is interdisciplinary for instance
					doc['tweet_type'] = mode
					# save the data of the tweet
					doc['tweet_date'] = datetime.strptime(re.sub(r'[+-]([0-9])+', '', tweet['created_at']),'%a %b %d %H:%M:%S %Y')
					# save the content of the tweet (this is the full raw content, we will parse out certain fields later on)
					doc['tweet_raw'] = tweet

					# save document to database
					db.insert_one_to_collection(collection = 'raw_tweets', doc = doc)

				else:
					logging.info('Tweet ID {} already processed, skipping...'.format(tweet['id']))
		
