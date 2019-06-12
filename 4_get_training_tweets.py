# -*- coding: utf-8 -*-

"""
	Created by:	Shaheen Syed
	Date: 		July 2018

	Step 4 - Get Training Tweets
	----------------------------
	
	This script uses labeled tweets that will serve as training tweets to create a machine learning classifier. Here we utilize labeled datasets from online repositories. Such labeled datasets have been labeled by human annotators for positive, negative, and neutral sentiment class. Note that the Twitter terms of service do not permit direct distribution of tweet content and so tweet IDs (references to the original tweets), with their respective sentiment labels, are often made available without the original tweet text and associated meta-data. These datasets can be found in the folder 'files/training_tweets'. As a consequence, we will have to use the Twitter API to retrieve the full tweet content, the tweet text, and the meta-data, by searching for the tweet ID. Some tweets will appear not to be available from the Twitter API and this, in some cases, results in the training datasets having fewer tweets than originally included in the published datasets.

	The description of the datasets can be found below.

	### What do the switches do

	There are 7 switches that can be turned on or off (by setting the value to True or False). Each switch will collect the tweets from the Twitter API from a specific training dataset.

	*	get_sanders_tweets = [True|False]
	*	get_semeval_tweets = [True|False]
	*	get_clarin13_tweets = [True|False]
	*	get_hcr_tweets = [True|False]
	*	get_omd_tweets = [True|False]
	*	get_stanford_test_tweets = [True|False]
	*	get_manual_labeled_tweets = [True|False]

	How to run:
	python 4_get_training_tweets.py

"""

# packages and modules
import json
import math # some special math operations
from collections import Counter # to get frequencies of items in list
from helper_functions import *
from database import MongoDatabase
from twitter import Twitter

# Twitter API keys
API_KEY = ''
API_SECRET = ''

# switches
get_sanders_tweets = False
get_semeval_tweets = False
get_clarin13_tweets = False
get_hcr_tweets = False
get_omd_tweets = False
get_stanford_test_tweets = False
get_manual_labeled_tweets = True


"""
	Internal Helper Functions
"""
def get_tweet_by_id(tweet_id):

	"""
		Call Twitter API and return tweet from tweet ID

		Parameters
		---------
		tweet_id: string
			unique tweet ID

		Returns
		--------
		new_doc : dictionary
			new document that can be inserted into the database

	"""

	# extract full tweet
	tweet = twitter.get_status(id = tweet_id, tweet_mode = 'extended')

	# convert to json
	if tweet is not None:
		tweet = json.dumps(tweet._json)

	return tweet


"""
	Script starts here
"""

if __name__ == "__main__":

	# create logging to console
	set_logger()

	logging.info('Start: {} '.format(__file__))

	# create database connection
	db = MongoDatabase()

	# instantiate twitter object
	twitter = Twitter(key = API_KEY, secret = API_SECRET)

	# create connection
	twitter.connect_to_API()

	if get_sanders_tweets:

		"""
			The Sanders dataset consists out of 5,513 hand classified tweets related to the topics Apple (@Apple), Google (#Google), Microsoft (#Microsoft), and Twitter (#Twitter). Tweets were 
			classified as positive, neutral, negative, or irrelevant; the latter referring to non-English tweets which we discarded. The Sanders dataset has been used for boosting Twitter 
			sentiment classification using different sentiment dimensions, combining automatically and hand-labeled twitter sentiment labels, and combining community detection and sentiment 
			analysis}. The dataset is available from http://www.sananalytics.com/lab/.

			Tweets are saved into the collection 'sanders_tweets_raw'
		"""	

		# name of the collection to store tweets to
		db_collection = 'sanders_tweets_raw'

		# location sanders tweets
		sanders_tweets_location = os.path.join('files', 'training_tweets', 'sanders', 'sanders_tweets.csv')

		# read sanders tweets
		data = read_csv(sanders_tweets_location)

		# read tweets that have already been processed (if you run for the first time, this will be an empty set)
		processed_tweets = set([x['tweet_id'] for x in db.read_collection(collection = db_collection)])

		# loop over each row of the CSV file
		for i, row in enumerate(data):

			# verbose
			logging.info('Processing tweet {}/{}'.format(i + 1, len(data)))

			# get values from columns
			tweet_label = row[1]
			tweet_id = row[2]

			# check if tweet_id has already been processed
			if not tweet_id in processed_tweets:

				# get content of the tweet
				tweet = get_tweet_by_id(tweet_id)

				# create new document to insert into the database
				new_doc = {}
				# add label
				new_doc['label'] = tweet_label
				# add tweet id
				new_doc['tweet_id'] = tweet_id
				# add raw tweet content
				new_doc['tweet'] = tweet

				# insert into database
				db.insert_one_to_collection(collection = db_collection, doc = new_doc)
	
	# execute if set to True
	if get_semeval_tweets:

		"""
			The Semantic Analysis in Twitter Task 2016 dataset, also known as SemEval-2016 Task 4, was created for various sentiment classification tasks. The tasks can be seen as challenges where 
			teams can compete amongst a number of sub-tasks, such as classifying tweets into positive, negative and neutral sentiment, or estimating distributions of sentiment classes. 
			Typically, teams with better classification accuracy or other performance measure rank higher. The dataset consist of training, development, and development-test data that combined 
			consist of 3,918 positive, 2,736 neutral, and 1,208 negative tweets. The original dataset contained a total of 10,000 tweets -- 100 tweets from 100 topics. Each tweet was labeled 
			by 5 human annotators and only tweets for which 3 out of 5 annotators agreed on their sentiment label were considered. The dataset is available from http://alt.qcri.org/semeval2016/task4/.

			Tweets are saved into the collection 'semeval_tweets_raw'
		"""

		# name of the collection to store tweets to
		db_collection = 'semeval_tweets_raw'

		# location semeval tweets
		semeval_train_tweets_location = os.path.join('files', 'training_tweets', 'semeval', 'train.csv')
		semeval_test_tweets_location = os.path.join('files', 'training_tweets', 'semeval', 'test.csv')
		semeval_dev_tweets_location = os.path.join('files', 'training_tweets', 'semeval', 'dev.csv')

		# read CSV
		data = read_csv(semeval_train_tweets_location) + read_csv(semeval_test_tweets_location) + read_csv(semeval_dev_tweets_location)
	
		# read tweets that have already been processed (if you run for the first time, this will be an empty set)
		processed_tweets = set([x['tweet_id'] for x in db.read_collection(collection = db_collection)])

		# loop over each row in the list
		for i, row in enumerate(data):

			# verbose
			logging.info('Processing tweet {}/{}'.format(i + 1, len(data)))

			# get values from columns
			tweet_label = row[1]
			tweet_id = row[0]

			# check if tweet_id has already been processed
			if not tweet_id in processed_tweets:

				# get content of the tweet
				tweet = get_tweet_by_id(tweet_id)

				# create new document to insert into the database
				new_doc = {}
				# add label
				new_doc['label'] = tweet_label
				# add tweet id
				new_doc['tweet_id'] = tweet_id
				# add raw tweet content
				new_doc['tweet'] = tweet

				# insert into database
				db.insert_one_to_collection(collection = db_collection, doc = new_doc)

	# execute if set to True
	if get_clarin13_tweets:

		"""
			The CLARIN 13-languages dataset contains a total of 1.6 million labeled tweets from 13 different languages, the largest sentiment corpus made publicly available. We used the English subset of the dataset since we restricted 
			our analysis to English tweets. Tweets were collected in September 2013 by using the Twitter Streaming API to obtain a random sample of 1% of all publicly available tweets. The tweets were manually annotated by assigning a 
			positive, neutral, or negative label by a total of 9 annotators; some tweets were labeled by more than 1 annotator or twice by the same annotator. For tweets with multiple annotations, only those with two-third agreement 
			were kept. The original English dataset contained around 90,000 labeled tweets. After recollection, a total of 15,064 positive, 24,263 neutral, and 12,936 negative tweets were obtained. The dataset is available 
			from http://hdl.handle.net/11356/1054.

			Tweets are saved into the collectin 'clarin13_tweets_raw'
		"""

		# name of the collection to store tweets to
		db_collection = 'clarin13_tweets_raw'

		# location clarin-13 tweets
		clarin13_tweets_location = os.path.join('files', 'training_tweets', 'clarin13', 'English_Twitter_sentiment.csv')

		# read CSV and skip the first row (its the header)
		data = read_csv(clarin13_tweets_location)[1:]

		# read tweets that have already been processed (if you run for the first time, this will be an empty set)
		processed_tweets = set([x['tweet_id'] for x in db.read_collection(collection = db_collection)])

		# loop over each row in the list
		for i, row in enumerate(data):

			# verbose
			logging.info('Processing tweet {}/{}'.format(i + 1, len(data)))

			# get values from columns
			tweet_label = row[1].lower()
			tweet_id = row[0]

			# check if tweet_id has already been processed
			if not tweet_id in processed_tweets:

				# get content of the tweet
				tweet = get_tweet_by_id(tweet_id)

				# create new document to insert into the database
				new_doc = {}
				# add label
				new_doc['label'] = tweet_label
				# add tweet id
				new_doc['tweet_id'] = tweet_id
				# add raw tweet content
				new_doc['tweet'] = tweet

				# insert into database
				db.insert_one_to_collection(collection = db_collection, doc = new_doc)


	if get_hcr_tweets:
		
		"""
			The Health Care Reform (HCR) dataset was created in 2010 -- around the time the health care bill was signed in the United States -- by extracting tweets with 
			the hashtag #hcr. The tweets were manually annotated by the authors by assigning the labels positive, negative, neutral, unsure, or irrelevant. The dataset was 
			split into training, development and test data. We combined the three different datasets that contained a total of 537 positive, 337 neutral, and 886 negative tweets. 
			The tweets labeled as irrelevant or unsure were not included. The HCR dataset was used to improve sentiment analysis by adding semantic features to tweets. 
			The dataset is available from https://bitbucket.org/speriosu/updown.
	
			Tweets are saved into the collectin 'hcr_tweets_raw'
		"""

		# name of the collection to store tweets to
		db_collection = 'hcr_tweets_raw'

		# location hcr tweets
		hcr_train_tweets_location = os.path.join('files', 'training_tweets', 'hcr', 'hcr-train.csv')
		hcr_test_tweets_location = os.path.join('files', 'training_tweets', 'hcr', 'hcr-test.csv')
		hcr_dev_tweets_location = os.path.join('files', 'training_tweets', 'hcr', 'hcr-dev.csv')

		# read CSV and skip the first row (its the header)
		data = read_csv(hcr_train_tweets_location)[1:] + read_csv(hcr_test_tweets_location)[1:] + read_csv(hcr_dev_tweets_location)[1:]

		# some rows have no label so we need to filter them out
		data = [x for x in data if x[1] != '']

		# read tweets that have already been processed (if you run for the first time, this will be an empty set)
		processed_tweets = set([x['tweet_id'] for x in db.read_collection(collection = db_collection)])

		# loop over each row in the list
		for i, row in enumerate(data):

			# verbose
			logging.info('Processing tweet {}/{}'.format(i + 1, len(data)))

			# get values from columns
			tweet_label = row[1].lower().strip()
			tweet_id = row[0]

			# check if tweet_id has already been processed
			if not tweet_id in processed_tweets:

				# get content of the tweet
				tweet = get_tweet_by_id(tweet_id)

				# create new document to insert into the database
				new_doc = {}
				# add label
				new_doc['label'] = tweet_label
				# add tweet id
				new_doc['tweet_id'] = tweet_id
				# add raw tweet content
				new_doc['tweet'] = tweet

				# insert into database
				db.insert_one_to_collection(collection = db_collection, doc = new_doc)

	if get_omd_tweets:

		"""
			The Obama-McCain Debate (OMD) dataset contains 3,238 tweets collected in September 2008 during the United States presidential debates between Barack Obama and John McCain. The tweets 
			were collected by querying the Twitter API for the hash tags #tweetdebate, #current, and #debate08. A minimum of three independent annotators rated the tweets as positive, negative, 
			mixed, or other. Mixed tweets captured both negative and positive components. Other tweets contained non-evaluative statements or questions. We only included the positive and negative 
			tweets with at least two-thirds agreement between annotators ratings; mixed and other tweets were discarded. The OMD dataset has been used for sentiment classification by 
			social relations, polarity classification, and sentiment classification utilizing semantic concept features. The dataset is available from https://bitbucket.org/speriosu/updown.

			Rating Codes: 1 = negative, 2 = positive, 3 = mixed, 4 = other

			Tweets are saved into the collection 'omd_tweets_raw'
		"""

		# name of the collection to store tweets to
		db_collection = 'omd_tweets_raw'

		# location hcr tweets
		omd_tweets_location = os.path.join('files', 'training_tweets', 'omd', 'debate.csv')

		# read CSV 
		data = read_csv(omd_tweets_location)

		# read tweets that have already been processed (if you run for the first time, this will be an empty set)
		processed_tweets = set([x['tweet_id'] for x in db.read_collection(collection = db_collection)])

		# loop over each row in the list
		for i, row in enumerate(data):

			# verbose
			logging.info('Processing tweet {}/{}'.format(i + 1, len(data)))

			# get the tweet ID
			tweet_id = row[0]

			# check if tweet_id has already been processed
			if not tweet_id in processed_tweets:

				# We only included the positive and negative tweets with at least two-thirds agreement between annotators ratings; mixed and other tweets were discarded
				labels = [x for x in row[1:] if x != ""]
				labels = ['negative' if x == '1' else x for x in labels]
				labels = ['positive' if x == '2' else x for x in labels]

				# get label and count of rating that occured most
				tweet_label, count = Counter(labels).most_common(1)[0]
				# only include if at least 2/3 of the ratings (so positive, postive, negative means that we include the tweet as a positive tweet, whereas positive, negative will be excluded)
				if count >= math.ceil(len(labels) * 0.66):
					# only add the positive and negative tweets, ignore tweets that have been labeled as mixed or other
					if tweet_label in ['positive', 'negative']:

						# get content of the tweet
						tweet = get_tweet_by_id(tweet_id)

						# create new document to insert into the database
						new_doc = {}
						# add label
						new_doc['label'] = tweet_label
						# add tweet id
						new_doc['tweet_id'] = tweet_id
						# add raw tweet content
						new_doc['tweet'] = tweet

						# insert into database
						db.insert_one_to_collection(collection = db_collection, doc = new_doc)

	# execute if set to True
	if get_stanford_test_tweets:

		"""
			The Stanford Test dataset contains 182 positive, 139 neutral, and 177 negative annotated tweets. The tweets were labeled by a human annotator and were retrieved by querying the Twitter 
			search API with randomly chosen queries related to consumer products, company names and people. The Stanford Training dataset, in contrast to the Stanford Test dataset, contains 1.6 million 
			labeled tweets. However, the 1.6 million tweets were automatically labeled, thus without a human annotator, by looking at the presence of emoticons. For example, tweets that contained 
			the positive emoticon :-) would be assigned a positive label, regardless of the remaining content of the tweet. Similarly, tweets that contained the negative emoticon :-( would be assigned a 
			negative label. Such an approach is highly biased and we choose not to include this dataset for the purpose of creating a sentiment classifier from labeled tweets. The Stanford Test dataset, 
			although relatively small, has been used to analyze and represent the semantic content of a sentence for purposes of classification or generation, semantic smoothing to alleviate data sparseness 
			problem for sentiment analysis, and sentiment detection of biased and noisy tweets. The dataset is available from http://www.sentiment140.com/.
			
			first column provides the sentiment label
			'0' = negative
			'2'	= neutral
			'4' = positive

			Tweets are saved into the collection 'stanford_tweets_raw'
		"""

		# name of the collection to store tweets to
		db_collection = 'stanford_tweets_raw'

		# location hcr tweets
		stanford_test_tweets_location = os.path.join('files', 'training_tweets', 'stanford', 'test.csv')

		# read CSV 
		data = read_csv(stanford_test_tweets_location)

		# read tweets that have already been processed (if you run for the first time, this will be an empty set)
		processed_tweets = set([x['tweet_id'] for x in db.read_collection(collection = db_collection)])

		# loop over each row in the list
		for i, row in enumerate(data):

			# verbose
			logging.info('Processing tweet {}/{}'.format(i + 1, len(data)))

			# get the tweet label (search for the column with an x)
			tweet_label = get_sentiment_label(get_sentiment_code(row[0]))

			# get tweet ID
			tweet_id = row[1]

			# check if tweet_id has already been processed
			if not tweet_id in processed_tweets:

				# get content of the tweet
				tweet = get_tweet_by_id(tweet_id)

				# create new document to insert into the database
				new_doc = {}
				# add label
				new_doc['label'] = tweet_label
				# add tweet id
				new_doc['tweet_id'] = tweet_id
				# add raw tweet content
				new_doc['tweet'] = tweet

				# insert into database
				db.insert_one_to_collection(collection = db_collection, doc = new_doc)


	# execute if set to True
	if get_manual_labeled_tweets:

		"""
			A random subset of target tweets (tweets referring to the three modes of research) that were manually labeled. Adding a subset of target tweets to the training data allows for more accurate classification predictions
			since the words and phrases found within the target tweets might not necessarily exist in the training tweets that we used from various online repositories.

			Tweets are saved into the collection 'manual_tweets_raw'
		"""

		# name of the collection to store tweets to
		db_collection = 'manual_tweets_raw'

		# location of the manual labeled tweets
		manual_labeled_tweets_location = os.path.join('files', 'training_tweets', 'manual_labeled', 'labels.csv')

		# read CSV and skip the first row (its the header)
		data = read_csv(manual_labeled_tweets_location)[1:]

		# read tweets that have already been processed (if you run for the first time, this will be an empty set)
		processed_tweets = set([x['tweet_id'] for x in db.read_collection(collection = db_collection)])

		# loop over each row in the list
		for i, row in enumerate(data):

			# verbose
			logging.info('Processing tweet {}/{}'.format(i + 1, len(data)))

			# get the tweet label (search for the column with an x)
			if row[2] == 'x':
				tweet_label = 'negative'
			elif row[3] == 'x':
				tweet_label = 'neutral'
			elif row[4] == 'x':
				tweet_label = 'positive'
			else:
				logging.error('Tweet has no label, skipping...')
				continue

			# get tweet ID
			tweet_id = row[0]

			# check if tweet_id has already been processed
			if not tweet_id in processed_tweets:

				# get content of the tweet
				tweet = get_tweet_by_id(tweet_id)

				# create new document to insert into the database
				new_doc = {}
				# add label
				new_doc['label'] = tweet_label
				# add tweet id
				new_doc['tweet_id'] = tweet_id
				# add raw tweet content
				new_doc['tweet'] = tweet

				# insert into database
				db.insert_one_to_collection(collection = db_collection, doc = new_doc)

