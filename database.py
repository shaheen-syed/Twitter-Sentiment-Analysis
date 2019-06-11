# -*- coding: utf-8 -*-

"""
	Created by:	Shaheen Syed
	Date: 		August 2018

	Class that handles all the database actions
"""

# packages and modules
from pymongo import MongoClient
import time, logging, sys
from bson.objectid import ObjectId


class MongoDatabase:

	def __init__(self, client = 'twitter'):

		self.client = MongoClient()
		self.db = self.client[client]


	def read_collection(self, collection):

		"""
			Read all documents in a certain collection
		"""

		try:
			return self.db[collection].find({}, no_cursor_timeout = True)
		except Exception, e:
			logging.error("[{}] : {}".format(sys._getframe().f_code.co_name,e))
			exit(1)

	def insert_one_to_collection(self, collection, doc):


		"""
			Insert one document to a collection
		"""

		try:
			self.db[collection].insert_one(doc)
		except Exception, e:
			logging.error("[{}] : {}".format(sys._getframe().f_code.co_name,e))
			exit(1)


	def update_collection(self, collection, doc):


		"""
			Update document to a collection
		"""

		try:	
			self.db[collection].update({'_id' : ObjectId(doc['_id'])},
									doc,
									upsert = False)
		except Exception, e:
			logging.error("[{}] : {}".format(sys._getframe().f_code.co_name,e))
			exit(1)
