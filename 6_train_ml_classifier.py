# -*- coding: utf-8 -*-

"""
	Created by:	Shaheen Syed
	Date: 		July 2018

	Step 6 - Train Several Machine Learning Classifiers
	---------------------------------------------------

	This script will train various machine learning classifiers to predict positive, negative, and neutral sentiment classes on the training tweets. The created ML models can then 
	be used to predict sentiment classes on the target tweets. The script can easily be adjusted to allow for other machine learning classifiers and parameter/hyper-parameter 
	values for grid-search. 

	How to run:
	python 6_train_ml_classifier.py

"""

# packages and modules
import json
import numpy as np
import random
from multiprocessing import cpu_count
from helper_functions import *
from database import MongoDatabase
# packages and modules for machine learning
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
random.seed(42)

"""
	Internal Helper Functions
"""

def get_pipeline_setup():

	"""
		Returns settings for the machine learning pipeline

		Returns
		-------
		pipeline_setup : dict()
			dictionary with ML classifier + pipeline options

	"""

	return { 'LinearSVC' 				: 	[{	
												'vectorizer' : True,
												'vectorizer_order' : -1,
												}],
			'SVC'						: [{	
												'vectorizer' : True,
												'vectorizer_order' : -1,
											}],
			'LogisticRegression'		: [{	
												'vectorizer' : True,
												'vectorizer_order' : -1,
										}],
			'DecisionTreeClassifier'	: [{	
												'vectorizer' : True,
												'vectorizer_order' : -1,
										}],
			'AdaBoostClassifier' 		: [{	
												'vectorizer' : True,
												'vectorizer_order' : -1,
										}],}

def get_grid_setup():

	"""	
		Return ML models and hyper-parameter values for grid search

		Returns
		-------
		grid_setup: dict()
			dictionary with ML models and parameter values

	"""
	return  {	
				'LinearSVC' : [{'hyperparameter' : 'C', 
								'random' : True, 
								'min' : 0.001, 
								'max' : 2.000, 
								'length' : 25},
								
								{'hyperparameter' : 'dual'},
								
								{'hyperparameter' : 'fit_intercept'},
								],

				
				'SVC' : 	[	{'hyperparameter' : 'C', 
								'random' : True, 
								'min' : 0.001, 
								'max' : 5.000, 
								'length' : 25 },

								{'hyperparameter' : 'gamma', 
								'random' : True,
								'min' : 0.001, 
								'max' : 1.00, 
								'length' : 25 },
								
								{'hyperparameter' : 'shrinking'},
								
								{'hyperparameter' : 'kernel'},
							],

				'LogisticRegression': [{'hyperparameter' : 'C',
										'random' : True,
										'min' : 0.001,
										'max' : 3.000,
										'length' : 500},
								
										{'hyperparameter' : 'dual'},
								
										{'hyperparameter' : 'fit_intercept'},
										],

				'DecisionTreeClassifier' :	[{'hyperparameter' : 'criterion'},
								
											{'hyperparameter' : 'splitter'},
								
											{'hyperparameter' : 'max_features'}
											],

				'AdaBoostClassifier' :	[	{'hyperparameter' : 'n_estimators',
												'random' : True,
												'min' : 1,
												'max' : 500,
												'length' : 200},
								
											{'hyperparameter' : 'algorithm'},
								
											{'hyperparameter' : 'adaboost_learning_rate',
											 'min' : 1,
											 'max' : 2}
										,]}

def create_parameter_grid(grid_setup):

	"""

		Creates the parameter grid dictionary to will be passed into the scikit learn gridsearchcv method
		
		Example:
			grid_setup = {	'LinearSVC' : [{'hyperparameter' : 'C', 
								'random' : False, 
								'min' : -5, 
								'max' : 10, 
								'length' : 100,
								'scaler' : True
								'scaler_order' : 1}],
				
				'SVC' : 	[	{'hyperparameter' : 'C', 
								'random' : False, 
								'min' : -5, 
								'max' : 10, 
								'length' : 100 },

								{'hyperparameter' : 'gamma', 
								'random' : True,
								'min' : -5, 
								'max' : 10, 
								'length' : 100 },

								{'hyperparameter' : 'sublinear_ngram_range',
								'min' : 1,
								'max' : 4}
							] 
				}
	"""

	# empty parameter grid dictionary
	param_grid = {}

	# loop over the rows
	for row in grid_setup:

		if row['hyperparameter'] == 'C':
			param_grid.update({'classify__C': get_gridsearch_number_between(row['random'], row['min'], row['max'], row['length']) })
		
		if row['hyperparameter'] == 'gamma':
			param_grid.update({'classify__gamma': get_gridsearch_number_between(row['random'], row['min'], row['max'], row['length']) })
		
		if row['hyperparameter'] == 'alpha':
			param_grid.update({'classify__alpha': get_gridsearch_number_between(row['random'], row['min'], row['max'], row['length']) })
		
		if row['hyperparameter'] == 'fit_prior':
			param_grid.update({'classify__fit_prior' : [True, False]})

		if row['hyperparameter'] == 'dual':
			param_grid.update({'classify__dual' : [True, False]})

		if row['hyperparameter'] == 'loss':
			param_grid.update({'classify__loss' : ['hinge', 'squared_hinge']})

		if row['hyperparameter'] == 'fit_intercept':
			param_grid.update({'classify__fit_intercept' : [True, False]})

		if row['hyperparameter'] == 'fit_shrinking':
			param_grid.update({'classify__shrinking' : [True, False]})

		if row['hyperparameter'] == 'kernel':
			param_grid.update({'classify__kernel' : ['linear', 'poly', 'rbf', 'sigmoid']})

		"""
			DESCISION TREES
		"""

		if row['hyperparameter'] == 'criterion':
			param_grid.update({'classify__criterion' : ['gini','entropy']})

		if row['hyperparameter'] == 'splitter':
			param_grid.update({'classify__splitter' : ['best','random']})

		if row['hyperparameter'] == 'max_features':
			param_grid.update({'classify__max_features' : ['auto','sqrt', 'log2', None]})

		if row['hyperparameter'] == 'n_estimators':
			param_grid.update({'classify__n_estimators' : [int(x) for x in get_gridsearch_number_between(row['random'], row['min'], row['max'], row['length'])]})

		if row['hyperparameter'] == 'algorithm':
			param_grid.update({'classify__algorithm' : ['SAMME','SAMME.R']})

		if row['hyperparameter'] == 'adaboost_learning_rate':
			param_grid.update({'classify__learning_rate' : range(row['min'], row['max'])})			


		"""
			VECTORIZER
		"""
		if row['hyperparameter'] == 'sublinear_tf':
			param_grid.update({'vectorizer__sublinear_tf': [True, False]})
		
		if row['hyperparameter'] == 'sublinear_use_idf':
			param_grid.update({'vectorizer__use_idf': [True, False]})
		
		if row['hyperparameter'] == 'sublinear_ngram_range':
			param_grid.update({'vectorizer__ngram_range': zip(np.repeat(row['min'], row['max'] + 1), range(row['min'], row['max'] + 1))})

		if row['hyperparameter'] == 'min_df':
			param_grid.update({'vectorizer__min_df': get_gridsearch_number_between(row['random'], row['min'], row['max'], row['length']) })
	
		if row['hyperparameter'] == 'max_df':
			param_grid.update({'vectorizer__max_df': get_gridsearch_number_between(row['random'], row['min'], row['max'], row['length']) })

			
	return param_grid

def get_classifier(classifier):

	"""
		Return the appropriate classifier method that scikit learn uses

		Parameters
		-----------
		classifier : string
			the name of the classifier, basically the name of the method in string

		Returns:
		classifier : scikit learn method
	"""

	if classifier == 'SVC':
		return svm.SVC(max_iter=1000000)
	elif classifier == 'LinearSVC':
		return svm.LinearSVC(max_iter=1000000)
	elif classifier == 'LogisticRegression':
		return LogisticRegression(max_iter=1000000)
	elif classifier == 'MultinomialNB':
		return MultinomialNB()
	elif classifier == 'BernoulliNB':
		return BernoulliNB()
	elif classifier == 'DecisionTreeClassifier':
		return DecisionTreeClassifier()
	elif classifier == 'AdaBoostClassifier':
		return AdaBoostClassifier()
	else:
		logging.error('Classifier {} not part of classifier list'.format(classifier))
		exit(1)


def get_gridsearch_number_between(random, start, end, length = 1):

	"""
		Get an array of floats between two values (start and stop) random or a structured range

		Parameters
		----------
		random : Boolean
			return random values or range
		start: int/float
			the start of the range
		end: int/float
			the end of the range
		length: int (optional)
			the number of values to generate

		Returns
		--------
		random_grid_search_values : np.array()
			numpy array of floats between start and end of length=length

	"""

	if random:
		return [np.random.uniform(start, end) for x in range(0,length)]
	else:
		return np.linspace(start, end, length)


def create_pipeline(classifier, pipeline_setup):

	"""
		Dynamically create the pipeline

		Parameters
		----------
		classifier : string
			name of the classifier algorithm to use
		pipeline_setup : dictionary
			additional options for the pipeline, such as for instance a vectorizer
	"""


	# create pipeline with classifier
	pipeline = Pipeline([('classify', get_classifier(classifier))])

	# add processes to the pipeline
	for row in pipeline_setup:

		if row.get('vectorizer'):
			pipeline.steps.insert(row['vectorizer_order'],['vectorizer', TfidfVectorizer()])
		
	return pipeline


def execute_gridsearch_cv(X, Y, test_size, shuffle, pipeline_setup, grid_setup, cv, n_jobs, scoring, verbose, save_model, model_save_location):

	"""
		Execute a grid search cross validated classification model creation

		Parameters
		----------
		X : np.array((n_samples, features))
			array with feature values
		Y : np.array((n_samples, 1))
			class label assignment
		test_size: float
			percentage of the test dataset (for instance 0.2 for 20% of the data for testing)
		shuffle : Boolean
			set to True if the training/test dataset creation need shuffling first
		pipeline_setup: dictionary
			settings for the pipeline, for instance, use scaling or not?
		grid_setup: dictionary
			settings for the grid search, for instance, the values for hyperparameters etc.
		cv: int
			number of fold within cross validation
		n_jobs: int
			number of parallel threads to create model
		scoring: string
			scoring function for cross validation, for isntance, f1_weighted
		verbose: int
			debug output, 10 is most
		save_model : Boolean
			if model needs to be saved to disk
		model_save_location: os.path
			if model needs to be saved, this is the location.
	"""

	try:

		# split data into train and test
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_size, random_state = 42, shuffle = shuffle)

		# execute gridsearch for each classifier
		for classifier in grid_setup.keys():

			# get parameter grid values
			param_grid = create_parameter_grid(grid_setup[classifier])

			# create pipeline
			pipeline = create_pipeline(classifier, pipeline_setup[classifier])


			# perform K-fold Cross validation
			grid_search = GridSearchCV(pipeline, cv = cv, n_jobs = n_jobs, param_grid = param_grid, scoring = scoring, verbose = verbose)

			# fit the grid_search model to the training data
			grid_search.fit(X_train,y_train)

			# calculate scores on test set
			precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(y_test, grid_search.predict(X_test), average='weighted')
			# get the confusion matrix
			cf_matrix = confusion_matrix(y_test, grid_search.predict(X_test))

			# create the results
			results = {'training_f1' : grid_search.best_score_,
						'best_parameters' : grid_search.best_params_,
						'test_precision' : precision,
						'test_recall' : recall,
						'test_f1' : f1,
						'test_confusion_matrix' : cf_matrix}

			if save_model:

				# make sure that the location to save the classifier to exists
				create_directory(model_save_location)

				# save classifier to disk
				joblib.dump(grid_search, os.path.join(model_save_location, classifier + '.pkl'))
				
				# or you can save with pickle, but then also load with pickle in subsequent steps of the analysis
				# save_pickle(obj = grid_search, file_name = classifier, folder = model_save_location)

				# save performance to disk
				save_dic_to_csv(results, file_name = classifier, folder = model_save_location)

	except Exception, e:
		logging.error('Error executing gridsearch CV: {}'.format(e))
		return

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

	# location to save machine learning classification models to
	model_save_location = os.path.join('files', 'ml_models2')

	# get all the training tweet documents
	D = db.read_collection(collection = db_collection)

	# get values from list and assign to X and Y
	X, Y = zip(*[(x['text'], str(x['label'])) for x in D])

	# define pipeline options
	pipeline_setup = get_pipeline_setup()

	# define grid search parameters and values
	grid_setup = get_grid_setup()

	# create and save the model
	execute_gridsearch_cv(X, Y, test_size = .2, shuffle = True, pipeline_setup = pipeline_setup, grid_setup = grid_setup, cv = 10, n_jobs = cpu_count(), scoring = 'f1_weighted', verbose = 10, save_model = True, model_save_location = model_save_location)
