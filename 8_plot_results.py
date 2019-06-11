# -*- coding: utf-8 -*-

"""
	Created by:	Shaheen Syed
	Date: 		July 2018

	Various plots
	- Donutplot showing sentiments per mode of research (int./trans./mult.)
	- Sentiment over time shown as a stacked bar chart per week
	- Sentiment by occuptation, only showing the most positive occupations
	- Bar chart showing the frequency of user tags, @, and URLs for each sentiment and each mode of research

"""

# packages and modules
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
from helper_functions import *
from database import MongoDatabase
from sklearn.externals import joblib


# Turn on/off what needs to be executed
create_donot_plot = True
create_time_stacked_bar_plot = True
create_sentiment_by_occupation = True
create_twitter_tokens_bar_plot = True

"""
	Script starts here
"""

if __name__ == "__main__":

	# create logging to console
	set_logger()

	# verbose
	logging.info('Start: {} '.format(__file__))

	# read labels
	labels = joblib.load(os.path.join('files', 'labels', 'labels.pkl'))

	# dictionary of with key = tweet_id and value = label
	tweet_id_to_label = {l[0]:l[1] for l in labels}

	# define the label types
	label_types = ['positive', 'neutral', 'negative']
	
	# define the colors for each sentiment (postive, neutral and negative)
	colors = ['#52bf80','#088bdc', '#fe6300']

	# plot location
	plot_location = os.path.join('files', 'plots')
	
	# create location if not exists
	create_directory(plot_location)

	# create database connection
	db = MongoDatabase()


	if create_donot_plot:

		"""
			Create a donutplot for interdisciplinary, transdisciplinary and multidisciplinary target tweets where each
			donutplot shows the percentage of positive, negative and neutral tweets

			file will be saved to files/plots/donutplot.pdf
		
		"""

		# create the subplts
		fig, axs = plt.subplots(1,3, figsize=(21, 6))
		# make axes available like ax[0] instead of ax[0,0]
		axs = axs.ravel()

		# loop over each axis and plot the donut
		for i, ax in enumerate(axs.reshape(-1)):

			# retrieve only rows for specific tweet type, e.g. interdisciplinary, multidisciplinary
			subset_labels = labels[labels[:,2] == i ]

			# set the plot title, for example, interdiscipline n = xxxx			
			ax.set_title(get_tweet_type_from_code(i) + ' n = ' + str(len(subset_labels)))

			# get number of tweets
			sizes = [
					len(subset_labels[subset_labels[:,1] == 2]),
					len(subset_labels[subset_labels[:,1] == 1]),
					len(subset_labels[subset_labels[:,1] == 0]),
					]

			# create the labels for the legend
			names = [
					'positive n = ' + str(sizes[0]), 
					'neutral n = '+ str(sizes[1]), 
					'negative n = '+ str(sizes[2])
					]

			# plot donut
			ax.pie(sizes, autopct='%.0f%%', colors = colors, startangle = 90, counterclock = False, wedgeprops = { 'linewidth' : 5, 'edgecolor' : 'white' })
			# add legend
			ax.legend(names,loc = "center", frameon = False)
			# nice round plots
			ax.axis('equal')
			my_circle = plt.Circle( (0,0), 0.7, color='white')
			ax.add_artist(my_circle)

		# adjust somewhat
		plt.subplots_adjust(wspace = 0.0, hspace=0.0)
		# remove some white space
		plt.tight_layout()
		# save figure
		fig.savefig(os.path.join(plot_location, 'donutplot.pdf'))
		# close plot so we can plot again if necessary
		plt.close()



	if create_time_stacked_bar_plot:

		"""
			Create a stacked bar chart that shows the sentiment over time, each bar shows the positive, negative and neutral tweets
		"""

		# get target tweet documents from database
		D = db.read_collection(collection = 'target_tweets')

		# create dictionary of week numbers per tweet id
		dic_weeks = {}
		for d in D:
			dic_weeks[d['tweet_id']] = '{}-{}'.format(d['tweet_date'].year, str(d['tweet_date'].isocalendar()[1]).zfill(2))

		# get list of year + week 
		weeks = sorted(set([x for x in dic_weeks.values()]))

		# some tweets from a part of week 31 were obtained because the api looks at 7 days history but we don't need them because we only want full weeks
		# weeks.remove('2017-31')

		# create the figure environment so we can plot the barcharts
		fig, axs = plt.subplots(3,1, figsize=(30, 19))
		axs = axs.ravel()

		# loop over the different tweet types
		for tweet_type in range(3):

			# get subset of labels
			subset_labels = labels[labels[:,2] == tweet_type]

			# create empty dataframe
			df = pd.DataFrame(index = pd.Series(label_types).values)
			
			# loop trough weeks and get sentiment values
			for week in weeks:

				week_tweet_ids = set()
				for key, value in dic_weeks.items():
					if value == week:
						if key in subset_labels[:,0]:
							week_tweet_ids.add(key)

				# get labels
				week_labels = []
				for i in week_tweet_ids:
					week_labels.append(tweet_id_to_label[i])

				# get the counts per label per week
				label_counts = Counter(week_labels)

				# create series
				df[week] = pd.Series([label_counts[2], label_counts[1], label_counts[0]]).values

			# tranpose dataframe
			df = df.transpose()
	
			# plot the stacked bar chart
			df.plot(kind = 'bar', stacked = True, color = colors, fontsize= 24, rot = 90, width = 0.8, linewidth = 0, alpha = 1., ax = axs[tweet_type])

			# omit the ticks on the first and second plot, so only show ticks on the bottom one
			if tweet_type != 2:
				axs[tweet_type].set_xticklabels([])

			# set the y labels
			axs[tweet_type].set_ylabel('Number of tweets', fontsize='24')
			# set the subplot title
			axs[tweet_type].set_title(get_tweet_type_from_code(tweet_type), fontsize='28')

		# set the legend and reverse the legend order
		for i in range(3):
			handles, legend_labels = axs[i].get_legend_handles_labels()
			axs[i].legend(handles[::-1], legend_labels[::-1],loc = "upper left", frameon = False, fontsize='24')
			
		# remove some white space
		plt.tight_layout()
		# save figure
		fig.savefig(os.path.join(plot_location, 'sentiment-over-time.pdf'))
		# close plot so we can plot again if necessary
		plt.close()



	if create_sentiment_by_occupation:

		"""
			Create a stacked bar chart with sentiment class by occupation
		"""

		# get target tweet documents from database
		D = db.read_collection(collection = 'target_tweets')

		# get counts of sentiment values per occupation
		dic_counts = {}

		# loop over all the tweet documents
		for i, d in enumerate(D):

			# get occupation matches
			matches = d['matches']
			
			# get inferred label 0 = negative, 1 = neutral, 2 = negative 
			label = d['label']
			# get label for sentiment (for example, the word 'positive')
			label_text = get_sentiment_label(label)
			# get the tweet type (int/transd/multid/)
			tweet_type = d['tweet_type']

			# check if tweet type is not part of dictionary, if not, add it
			if tweet_type not in dic_counts:
				dic_counts[tweet_type] = {}

			# loop over each of the matched occuptations
			for occupation in matches:

				# skip matches that are not an occuptation but mere a reference to academic setting
				if occupation.strip() in ['research', 'science', 'university', 'education', 'college', 'studies', 'sciences', 'scientific', 'faculty', 'academy', 'academics','mathematics', 'health science',
											'higher education', 'doctoral', 'researching', 'researchers', 'undergraduate','graduate', 'neuroscience', 'scientists', 'humanities', 'anthropology', 'health professional', 'business school']:
					continue

				# merge occuptations
				if occupation.strip() in ['phd', 'phd student', 'phd candidate', 'ph.d.']: occupation = 'phd candidate'
				if occupation.strip() in ['prof','prof.']: occupation = 'professor'
				if occupation.strip() in ['post-doc','post doc']: occupation = 'postdoc'

				# check if occupation is not part of the tweet type, if not, add it
				if occupation not in dic_counts[tweet_type]:
					dic_counts[tweet_type][occupation.strip()] = {'negative' : 0, 'neutral' : 0, 'positive' : 0}
				
				# add 1 to the sentiment label
				dic_counts[tweet_type][occupation][label_text] += 1

		# create the plot environment
		fig, axs = plt.subplots(3,1, figsize=(15, 15))
		axs = axs.ravel()

		# plot mode of research onto each axis
		for i, key in enumerate(['interdisciplinary', 'transdisciplinary','multidisciplinary']):

			# get the values from the dictionary
			values = dic_counts[key]

			# create empty dataframe
			df = pd.DataFrame()

			# loop over the values
			for occupation, counts in values.iteritems():
				
				# add counts per sentiment as series to dataframe
				df['{} (n={})'.format(occupation, sum(counts.values()))] = pd.Series(counts)

			# tranpose the dataframe	
			df = df.T

			# get a copy of the totals
			df['total'] = df.sum(axis=1)

			# calculate percentage of positive tweets in relation to total tweets for occupation
			df['positive-percentage'] = df['positive'] / df['total'] * 100. 
			
			# calculate percentage of negative tweets in relation to total tweets for occupation
			df['negative-percentage'] = df['negative'] / df['total'] * 100. 

			df_subset = df.sort_values(by = ['total'], ascending=False)[0:25][['positive-percentage', 'negative-percentage']].sort_values(by = ['positive-percentage'], ascending=False)

			# plot positive percentages
			df_subset.plot(kind = 'bar', stacked = True, color = ['#52bf80', '#fe6300'], fontsize= 14, rot = 45, width = 0.8, linewidth = 0, ax = axs[i])

			# alighn the x labels
			for xtick in axs[i].get_xticklabels():
				xtick.set_ha('right')

			# set y limit
			axs[i].set_ylim(0,75)
			# set y label
			axs[i].set_ylabel('% tweets', fontsize='14')
			# set title
			axs[i].set_title(key, fontsize='16')

		# set the legend and reverse the legend order
		for i in range(3):
			handles, _ = axs[i].get_legend_handles_labels()
			sentiment_labels = ['positive', 'negative']
			axs[i].legend(handles[::-1], sentiment_labels[::-1],loc = "upper right", frameon = False, fontsize='14')

		# remove some white space
		plt.tight_layout()
		# save figure
		fig.savefig(os.path.join(plot_location, 'sentiment-by-occupation.pdf'))
		# close plot so we can plot again if necessary
		plt.close()



	if create_twitter_tokens_bar_plot:

		"""
			Create a bar plot with frequence of emoji, @, and URL for each sentiment and for each mode of research
		"""

		# create dictionary with key = tweet_id and value = text
		id_to_text = {d['tweet_id'] : d['raw_text'] for d in db.read_collection( collection = 'target_tweets')}

		# empty list so we can add data to it
		data = []

		# loop over each tweet type, note that they are encoded as 0 = interdisciplinary, 1 = transdisciplinary, 2 = multidisciplinary
		for tweet_type in range(3):

			logging.info('Processing tweet type: {}'.format(tweet_type))

			# loop trough label type, note that they are encoded as negative = 0, neutral = 1 and positive = 2
			for label_type in range(3):

				logging.info('Processing label: {}'.format(label_type))

				# filter labels based on tweet type and label => interdisciplinary + positive
				subset_labels = labels[(labels[:,2] == tweet_type) & (labels[:,1] == label_type)][:,0]

				# counters for frequency of @, URL, and emoji
				num_at, num_url, num_emoji = 0,0,0

				# get only the tweets that are part of the subset
				subset_tweets = [id_to_text[x] for x in subset_labels]
				for text in subset_tweets:

					# count @
					num_at += text.count('@')
					# count url
					num_url += text.count('http')

					# replace emojis by placeholder
					text = replace_emojis(text, placeholder_pos = 'EMOJI', placeholder_neg = 'EMOJI')
					# count frequency of placeholder
					num_emoji += text.count('EMOJI')

				# get value to normalize total counts
				normalizer = float(len(subset_tweets))
				
				# add to data
				data.append([tweet_type, label_type, num_at / normalizer, num_url / normalizer, num_emoji / normalizer])


		# create the figure environment so we can plot the barcharts
		fig, axs = plt.subplots(1,3, figsize=(15, 5), sharey=True)
		axs = axs.ravel()

		# loop over each tweet type, note that they are encoded as 0 = interdisciplinary, 1 = transdisciplinary, 2 = multidisciplinary
		for i in range(3):

			# filter the data
			subset_data = np.array([x for x in data if x[0] == i])[:,1:]

			# create empty dataframe
			df = pd.DataFrame(index = reversed(label_types))
			# add at series
			df['@'] = pd.Series(subset_data[:,1]).values
			# add url series
			df['URL'] = pd.Series(subset_data[:,2]).values
			# add emoji series
			df['EMOJI'] = pd.Series(subset_data[:,3]).values

			# transpose the dataframe
			df = df.transpose()

			# plot the dataframe
			df.plot(kind = 'bar', stacked = False, color = reversed(colors), fontsize=12, rot = 0, width = 0.8, linewidth = 0, alpha = 1., ax = axs[i])

			# set the ylabel
			axs[i].set_ylabel('Frequency/#tweets', fontsize='12')
			# add the legend
			axs[i].legend(loc = "upper right", frameon = False, fontsize='12')
			# set the title
			axs[i].set_title(get_tweet_type_from_code(i), fontsize='12')
		
		# remove some white space
		plt.tight_layout()
		# save figure
		fig.savefig(os.path.join(plot_location, 'frequency-emoji-url-at.pdf'))
		# close plot so we can plot again if necessary
		plt.close()





