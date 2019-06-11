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
from helper_functions import *
from database import MongoDatabase


# switches, set to True what needs to be executed
filter_tweets = True
clean_tweets = True

"""
	Script starts here
"""

if __name__ == "__main__":

	# create logging to console
	set_logger()

	logging.info('Start: {} '.format(__file__))

	# create database connection
	db = MongoDatabase()

	# execute if set to True
	if filter_tweets:

		"""	
			Filter raw target tweet
				- remove non-English tweets
				- remove retweet
				- remove tweets that do  not originate from an academic or scientist (by using bio text)

			raw tweets are stored in the collection 'raw_tweets'
			filtered tweets will be stored in the collectin 'filtered_tweets'
		"""

		# read tweets documents from database
		D = db.read_collection(collection = 'raw_tweets')

		# tracker to keep track of processed tweet IDs (in case we want to repeat the process for a set of new tweets)
		tweet_tracker = set(['{}{}'.format(x['tweet_type'], x['id']) for x in db.read_collection( collection = 'filtered_tweets')] )

		# read academic/scientists professions (so we can filter the bio on these words)
		academic_words = [x.strip('\n').strip('\r').lower() for x in read_plain_text(os.path.join('files', 'filter_bio', 'academic_words.txt'), read_lines = True)]

		# loop over each tweet document
		for i, d in enumerate(D):

			logging.debug('Processing tweet {}/{}'.format(i + 1, D.count()))

			# check if doc already in database
			if not '{}{}'.format(d['tweet_type'], d['id']) in tweet_tracker:

				# get the raw tweet
				tweet = d['tweet_raw']
				# read the content of the tweet
				text = tweet['full_text']
				# read language
				lang = tweet['lang']
				# read user bio
				bio = tweet['user']['description'].lower().replace('\n', ' ').replace('\r', ' ')

				# Check for language as some tweets appear in non-english
				if lang != 'en':
					continue

				# check for retweets
				if text.startswith('RT '):
					continue

				# check if bio can be mapped to 1 or more academmic professions
				matches = []
				for w in academic_words:
					if w in bio:
						if ' bot ' not in bio:
							logging.info('Academic word match in bio: {}'.format(w))
							matches.append(w)

				if len(matches) == 0:
					continue

				# add matches so we can use it later
				d['matches'] = matches

				# remove _id so we can save it to database again but different collection
				del d['_id']

				# save doc to filtered_tweets collectino
				db.insert_one_to_collection(collection = 'filtered_tweets', doc = d)
			else:
				logging.debug('Tweet {} already processed'.format(d['id']))

	# execute if set to True
	if clean_tweets:

		"""
			Clean raw tweets that have already been filtered (if not filtered, set filter_tweets to True first)

			filtered tweets are stored in the collection 'filtered_tweets'

			After cleaning, tweets are stored in the collection 'target_tweets'

			Cleaning/preprocessing steps

				- replace new lines
				- replace ampersand character
				- replace @
				- replace URL
				- replace hashtag
				- replace contractions
				- replace emoticons
				- replace emojis
				- replace repeating charachtes : happyyyyy -> happyy
				- replace punctuation
				- replace specific characters
				- replace double spaced
				- trim leading and trailing spaces
				- tokenize
				- lemmatization
		"""

		# read tweets documents from database
		D = db.read_collection(collection = 'filtered_tweets')

		# tracker to keep track of processed tweet IDs (in case we want to repeat the process for a set of new tweets)
		tweet_tracker = set(['{}{}'.format(x['tweet_type'], x['tweet_id']) for x in db.read_collection( collection = 'target_tweets')] )

		# tracker for cleaned tweet content (so we can find duplicated content)
		tweet_text_tracker = set()

		# setup spacy object, so we can do some NLP things
		nlp = setup_spacy()

		# loop over each tweet document
		for i, d in enumerate(D):

			logging.debug('Processing tweet {}/{}'.format(i + 1, D.count()))

			# check if doc already in database
			if not '{}{}'.format(d['tweet_type'], d['id']) in tweet_tracker:

				# get the original tweet text
				raw_text = d['tweet_raw']['full_text']

				# preprocess the tweet text
				text = clean_tweet(raw_text)

				# tokenize the tweet text
				tokens = get_tokens(text)

				# convert to lemma
				tokens = get_lemma(nlp(' '.join(tokens)))

				# convert list to string again
				text = ' '.join(tokens).encode('utf-8')
				
				# create new document so we can insert it into the database
				new_doc = {}
				new_doc['tweet_id'] = d['id']
				new_doc['tweet_date'] = d['tweet_date']
				new_doc['tweet_type'] = d['tweet_type']
				new_doc['text'] = text
				new_doc['raw_text'] = raw_text
				new_doc['bio'] = d['tweet_raw']['user']['description'].lower().replace('\n', ' ').replace('\r', ' ')
				new_doc['matches'] = d['matches']

				# occasionally tweets will duplicate the tweet, even though they have different tweet IDs, they are the same and should not be included in the analysis
				# we utilize a somewhat crude way of finding duplicates, that is, the content of the cleaned tweet. The reason why we don't compare the raw content is that
				# the duplicated tweet often has a different URL at the end. The cleaning process will replace it into a URL placeholder, so it doesn't matter if they are different.
				if '{}{}'.format(d['tweet_type'], text) not in tweet_text_tracker:

					# add to tracker
					tweet_text_tracker.add('{}{}'.format(d['tweet_type'], text))

					# save doc to extended tweets
					db.insert_one_to_collection(collection = 'target_tweets', doc = new_doc)


