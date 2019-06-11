# -*- coding: utf-8 -*-

"""
	Created by:	Shaheen Syed
	Date: 		July 2018

	Use tweepy to connect to the Twitter API

"""

# packages and modules
import logging
import sys
import tweepy # to connect to twitter API

class Twitter:

	def __init__(self, key, secret):

		logging.info('Initialize {}'.format(self.__class__.__name__))

		# set key
		self.key = key
		
		# set secret
		self.secret = secret

		# set authentication
		self.auth = tweepy.AppAuthHandler(self.key, self.secret)


	def connect_to_API(self):

		"""
			Connect to the Twitter API by using the tweepy package
			This class provides a wrapper for the API as provided by Twitter. The functions provided in this class are listed below.

			More info : http://docs.tweepy.org/en/v3.5.0/api.html

			Parameters
			-----------
			auth_handler – authentication handler to be used
			host – general API host
			search_host – search API host
			cache – cache backend to use
			api_root – general API path root
			search_root – search API path root
			retry_count – default number of retries to attempt when error occurs
			retry_delay – number of seconds to wait between retries
			retry_errors – which HTTP status codes to retry
			timeout – The maximum amount of time to wait for a response from Twitter
			parser – The object to use for parsing the response from Twitter
			compression – Whether or not to use GZIP compression for requests
			wait_on_rate_limit – Whether or not to automatically wait for rate limits to replenish
			wait_on_rate_limit_notify – Whether or not to print a notification when Tweepy is waiting for rate limits to replenish
			proxy – The full url to an HTTPS proxy to use for connecting to Twitter.
		"""



		logging.info('Called function: {}.{} '.format(self.__class__.__name__,sys._getframe().f_code.co_name))

		self.api = tweepy.API(self.auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)

	def search_tweets(self, **kwarg):

		"""
			Use the Twitter Search API to collect tweets by using a search query
			- 	Returns tweets that match a specified query.

			Parameters
			-----------
			q – the search query string
			lang – Restricts tweets to the given language, given by an ISO 639-1 code.
			locale – Specify the language of the query you are sending. This is intended for language-specific clients and the default should work in the majority of cases.
			rpp – The number of tweets to return per page, up to a max of 100.
			page – The page number (starting at 1) to return, up to a max of roughly 1500 results (based on rpp * page.
			since_id – Returns only statuses with an ID greater than (that is, more recent than) the specified ID.
			geocode – Returns tweets by users located within a given radius of the given latitude/longitude. The location is preferentially taking from the Geotagging API, but will fall back to their Twitter profile. The parameter value is specified by “latitide,longitude,radius”, where radius units must be specified as either “mi” (miles) or “km” (kilometers). Note that you cannot use the near operator via the API to geocode arbitrary locations; however you can use this geocode parameter to search near geocodes directly.
			show_user – When true, prepends “<user>:” to the beginning of the tweet. This is useful for readers that do not display Atom’s author field. The default is false.
			
			Returns
			----------	
			list of SearchResult objects

		"""

		logging.info('Called function: {}.{} '.format(self.__class__.__name__,sys._getframe().f_code.co_name))

		try:
			# collect tweets
			return self.api.search(**kwarg)

		except tweepy.TweepError as e:
			logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
			exit(1)

	def get_status(self, **kwarg):

		"""
			Returns a single status specified by the ID parameter.

			Parameters
			-----------
			id – The numerical ID of the status.
			Tweet_mode:	|Pass in 'extended' to get non truncated tweet text|
			
			Returns
			--------
			Return type:	Status object

		"""

		try:
			# get status for single tweet
			return self.api.get_status(**kwarg)
		except tweepy.TweepError as e:
			logging.warning('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
			return None
