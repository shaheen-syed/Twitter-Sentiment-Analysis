# -*- coding: utf-8 -*-

"""
	Created by:	Shaheen Syed
	Date: 		July 2018

	Tweeting, the process of publishing a tweet, proceeds in the form of free text, often in combination with special characters, symbols, emoticons, and emoji. This, in combination with a character limit, make tweeters creative
	and concise in their writing, favoring brevity over readability to convey their message---even more so with the 140 characters limit. Thus tweet data is highly idiosyncratic and several preprocessing steps were necessary 
	(described below) to make the dataset suitable for sentiment analysis. 

		-	Retweets and duplicate tweets (only performed on target tweets)
		We removed retweets, identified by the string 'RT' preceding the tweet, as they essentially are duplicates of the initial or first tweet. Additionally, duplicate tweets that were identical in their content were also excluded.   

		-	Non-English tweets (only performed on target tweets)
		We focused our analysis on English tweets only and excluded all non-English tweets according to the 'lang' attribute provided by the Twitter API. 

		-	User tags and URLs
		For the purpose of sentiment analysis, the user tags (i.e., mentioning of other Twitter user accounts by using @) and URLs (i.e., a link to a specific website) convey no specific sentiment and were therefore replaced with a suitable placeholder 
		(e.g. USER, URL). As a result, the presence and frequency of user tags and URLs were retained and normalized.

		-	Hashtags
		Hashtags are an important element of Twitter and can be used to facilitate a search while simultaneously convey opinions or sentiments. For example, the hashtag #love reveals a positive sentiment or feeling, and tweets using the hashtag are 
		all indexed by #love. Twitter allows users to create their own hashtags and poses no restrictions in appending the hashtag symbol (i.e., #) in front of any given text. Following the example of the #love hashtag, we preprocessed hashtags 
		by removing the hash sign, essentially making #love equal to the word love. 

		-	Contractions and repeating characters
		Contractions, such as don't and can't, are a common phenomenon in the English spoken language and, generally, less common in formal written text. For tweets, contractions can be found in abundance and are an accepted means of communication. 
		Contractions were preprocessed by splitting them into their full two-word expressions, such as 'do not' and 'can not'. In doing so, we normalized contractions with their "decontracted" counterparts. Another phenomenon occurring in tweets 
		is the use of repeating characters, such as 'I loveeeee it', often used for added emphasis. Words that have repeated characters are limited to a maximum of two consecutive characters. For example, the word loveee and loveeee are normalized 
		to lovee. In doing so, we maintained some degree of emphasis.

		-	Lemmatization and uppercase words
		For grammatical reasons, different word forms or derivationally related words can have a similar meaning and, ideally, we would want such terms to be grouped together. For example, the words like, likes, and liked all have similar semantic 
		meaning and should, ideally, be normalized. Stemming and lemmatization are two NLP techniques to reduce inflectional and derivational forms of words to a common base form. Stemming heuristically cuts off derivational affixes to achieve 
		some kind of normalization, albeit crude in most cases. We applied lemmatization, a more sophisticated normalization method that uses a vocabulary and morphological analysis to reduce words to their base form, called lemma. 
		It is best described by its most basic example, normalizing the verbs am, are, is to be, although such terms are not important for the purpose of sentiment analysis. Additionally, uppercase and lowercase words were grouped as well.

		-	Emoticons and Emojis
		Emoticons are textual portrayals of a writer's mood or facial expressions, such as :-) and :-D (i.e., smiley face). For sentiment analysis, they are crucial in determining the sentiment of a tweet and should be retained within the 
		analysis. Emoticons that convey a positive sentiment, such as :-), :-], or ;), were replaced with the positive placeholder word; in essence, grouping variations of positive emoticons with a common word. Emoticons conveying a 
		negative sentiment, such as :-(, :c, or :-c, were replaced by the negative placeholder word. A total of 47 different variations of positive and negative emoticons were replaced. A similar approach was performed with emojis 
		that resemble a facial expression and convey a positive or negative sentiment. Emojis are graphical symbols that can represent an idea, concept or mood expression, such as the graphical icon of a happy face. A total of 40 emojis 
		with positive and negative facial expressions were replaced by a placeholder word. Replacing and grouping the positive and negative emoticons and emojis will result in the sentiment classification algorithm learning an 
		appropriate weight factor for the corresponding sentiment class. For example, tweets that have been labeled as conveying a negative sentiment (by a human annotator for instance) and predominantly containing negative 
		emoticons (e.g., :-(), can result in the classification algorithm assigning a higher probability or weight to the negative sentiment class for such emoticons. Note that this only holds when the neutral and positively 
		labeled tweets do not predominantly contain negative emoticons; otherwise their is no discriminatory power behind them.


		-	Numbers, punctuation, and slang
		Numbers and punctuation symbols were removed, as they typically convey no specific sentiment. Numbers that were used to replace characters or syllables of words were retained, such in the case of 'see you l8er'. We chose not to 
		convert slang and abbreviations to their full word expressions, such as brb for 'be right back' or 'ICYMI' for 'in case you missed it'. The machine learning model, described later, would correctly handle most common uses of slang, 
		with the condition that they are part of the training data. As a result, slang that is indicative of a specific sentiment class (e.g. positive or negative) would be assigned appropriate weights or probabilities during model creation.
"""

# packages and modules
import json
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

	# name of collection to store all the training tweets to
	db_collection = 'training_tweets'

	# setup spacy object, so we can do some NLP things
	nlp = setup_spacy()

	# sources to process and the collections they are stored in
	process_sources = {	'sanders' : 'sanders_tweets_raw',
						'semeval': 'semeval_tweets_raw',
						'clarin13': 'clarin13_tweets_raw',
						'hcr': 'hcr_tweets_raw',
						'omd': 'omd_tweets_raw',
						'stanford': 'stanford_tweets_raw',
						'manual': 'manual_tweets_raw'}

	# perform preprocessing on each training dataset
	for source, collection in process_sources.iteritems():

		logging.info('Processing tweets from source: {}'.format(source))

		# get all the raw tweets documents from collection
		D = db.read_collection(collection = collection)

		# loop over each of the tweet
		for i, d in enumerate(D):

			# verbose
			logging.debug('	-	Processing Tweet {}/{}'.format(i + 1, D.count()))

			# check if tweet could be extracted from the Twitter API (sometimes tweets are not available anymore when collecting them some time after they are created,
			# if this is the case, the content of tweet will be None)
			if d['tweet'] is not None:

				# check label type is pos, neg, or neu
				if d['label'] in ['positive', 'negative', 'neutral']:

					# convert tweet content to json
					tweet = json.loads(d['tweet'])
					# get the original tweet text
					raw_text = tweet['full_text']
					# preprocess the tweet text
					text = clean_tweet(raw_text)
					# tokenize the tweet text
					tokens = get_tokens(text)
	
					# check if at least 1 token
					if len(tokens) > 0:

						# convert to lemma
						tokens = get_lemma(nlp(' '.join(tokens)))
						# convert list to string again
						text = ' '.join(tokens).encode('utf-8')

						# create new document to insert into the database
						new_doc = {}
						# add tweet text
						new_doc['text'] = text
						# add raw text
						new_doc['raw_text'] = raw_text
						# add source
						new_doc['source'] = source
						# add label
						new_doc['label'] = get_sentiment_code(d['label'])

						# insert into database
						db.insert_one_to_collection(collection = db_collection, doc = new_doc)

	