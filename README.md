# Full workflow to perform sentiment analysis on Twitter data.

Code belonging to the following scientific paper:
Interdisciplinary Optimism? Sentiment Analysis of Twitter Data

It contains scripts to:
- search for tweets on which you want to perform sentiment analysis (referred to as target tweets)
- parsing of tweets collected from the Twitter API
- preprocessing of the tweets
- collect training tweets through the Twitter API that originate from several online repositories and are labeled by human annotators
- preprocessing of the training tweets
- create different machine learning classification models including grid search for hyper-parameter tuning
- infer sentiment class of the target tweets
- create various plots to show distribution of sentiment classes, distribution of classes per week, distribution of classes by occupation


# Description and purpose of the python files

## Step 1 – Search for target tweets

The tweets of interest are referred to as target tweets. That is, tweets for which we want to infer a sentiment class. In the paper, target tweets relate to tweets about interdisciplinarity, transdisciplinarity, and multidisciplinarity. This script uses the Twitter API to collect tweets that match a specific search query. Note that tweets are only available within the search API if not older than 7 days. To create a dataset, execute once every 7 days, either manually or by using something like a cronjob. The collected target tweets will be saved on disk. It will furthermore be used to only retrieve the delta of new tweets since the last time this script was run by reading the latest ID from the latest created file.

Before running, set the twitter key and secret, see https://developer.twitter.com/en/docs/basics/authentication/guides/access-tokens.html

How to run?
```
python 1_search_target_tweets.py
```

## Step 2 – Parse target tweets

This script reads all the .txt files created while running the script in step 1, and parses out the individual tweets and relevant fields. It then saves each tweet as a document in a MongoDB database. The script knows if tweets have already been inserted previously, so there is no need to check for this.

How to run:
```
python 2_parse_target_tweets.py
```


## Step 3 – Preprocess Target Tweets

Tweeting, the process of publishing a tweet, proceeds in the form of free text, often in combination with special characters, symbols, emoticons, and emoji. This, in combination with a character limit, make tweeters creative and concise in their writing, favoring brevity over readability to convey their message---even more so with the 140 characters limit. Thus tweet data is highly idiosyncratic and several preprocessing steps are necessary (described below) to make the dataset suitable for sentiment analysis. This script performs the following preprocessing steps that were also used in paper.

* Retweets and duplicate tweets (only performed on target tweets)
	-	We removed retweets, identified by the string 'RT' preceding the tweet, as they essentially are duplicates of the initial or first tweet. Additionally, duplicate tweets that were identical in their content were also excluded.   

* Non-English tweets (only performed on target tweets)
	-	We focused our analysis on English tweets only and excluded all non-English tweets according to the 'lang' attribute provided by the Twitter API. 

* User tags and URLs
	-	For the purpose of sentiment analysis, the user tags (i.e., mentioning of other Twitter user accounts by using @) and URLs (i.e., a link to a specific website) convey no specific sentiment and were therefore replaced with a suitable placeholder (e.g. USER, URL). As a result, the presence and frequency of user tags and URLs were retained and normalized.

* Hashtags
	-	Hashtags are an important element of Twitter and can be used to facilitate a search while simultaneously convey opinions or sentiments. For example, the hashtag #love reveals a positive sentiment or feeling, and tweets using the hashtag are all indexed by #love. Twitter allows users to create their own hashtags and poses no restrictions in appending the hashtag symbol (i.e., #) in front of any given text. Following the example of the #love hashtag, we preprocessed hashtags by removing the hash sign, essentially making #love equal to the word love. 

* Contractions and repeating characters
	-	Contractions, such as don't and can't, are a common phenomenon in the English spoken language and, generally, less common in formal written text. For tweets, contractions can be found in abundance and are an accepted means of communication. Contractions were preprocessed by splitting them into their full two-word expressions, such as 'do not' and 'can not'. In doing so, we normalized contractions with their "decontracted" counterparts. Another phenomenon occurring in tweets is the use of repeating characters, such as 'I loveeeee it', often used for added emphasis. Words that have repeated characters are limited to a maximum of two consecutive characters. For example, the word loveee and loveeee are normalized to lovee. In doing so, we maintained some degree of emphasis.

* Lemmatization and uppercase words
	-	For grammatical reasons, different word forms or derivationally related words can have a similar meaning and, ideally, we would want such terms to be grouped together. For example, the words like, likes, and liked all have similar semantic meaning and should, ideally, be normalized. Stemming and lemmatization are two NLP techniques to reduce inflectional and derivational forms of words to a common base form. Stemming heuristically cuts off derivational affixes to achieve some kind of normalization, albeit crude in most cases. We applied lemmatization, a more sophisticated normalization method that uses a vocabulary and morphological analysis to reduce words to their base form, called lemma. It is best described by its most basic example, normalizing the verbs am, are, is to be, although such terms are not important for the purpose of sentiment analysis. Additionally, uppercase and lowercase words were grouped as well.

* Emoticons and Emojis
	-	Emoticons are textual portrayals of a writer's mood or facial expressions, such as :-) and :-D (i.e., smiley face). For sentiment analysis, they are crucial in determining the sentiment of a tweet and should be retained within the analysis. Emoticons that convey a positive sentiment, such as :-), :-], or ;), were replaced with the positive placeholder word; in essence, grouping variations of positive emoticons with a common word. Emoticons conveying a negative sentiment, such as :-(, :c, or :-c, were replaced by the negative placeholder word. A total of 47 different variations of positive and negative emoticons were replaced. A similar approach was performed with emojis that resemble a facial expression and convey a positive or negative sentiment. Emojis are graphical symbols that can represent an idea, concept or mood expression, such as the graphical icon of a happy face. A total of 40 emojis with positive and negative facial expressions were replaced by a placeholder word. Replacing and grouping the positive and negative emoticons and emojis will result in the sentiment classification algorithm learning an appropriate weight factor for the corresponding sentiment class. For example, tweets that have been labeled as conveying a negative sentiment (by a human annotator for instance) and predominantly containing negative emoticons (e.g., :-(), can result in the classification algorithm assigning a higher probability or weight to the negative sentiment class for such emoticons. Note that this only holds when the neutral and positively labeled tweets do not predominantly contain negative emoticons; otherwise their is no discriminatory power behind them.

* Numbers, punctuation, and slang
	-	Numbers and punctuation symbols were removed, as they typically convey no specific sentiment. Numbers that were used to replace characters or syllables of words were retained, such in the case of 'see you l8er'. We chose not to convert slang and abbreviations to their full word expressions, such as brb for 'be right back' or 'ICYMI' for 'in case you missed it'. The machine learning model, described later, would correctly handle most common uses of slang, with the condition that they are part of the training data. As a result, slang that is indicative of a specific sentiment class (e.g. positive or negative) would be assigned appropriate weights or probabilities during model creation.

### What do the switches do:
There are two switches that can be turned on or off (by setting them to True or False).

*	filter_tweets = [True|False]
	-	exclude non-English tweets, exclude retweets, and matches words found in the bio field of the tweet to a list of occupations. Filtering for occupations enables the creation of a set of tweets from a specific audience (here originating from an academic setting)
*	clean_tweets = [True|False]
	-	performs the remaining of the preprocessing steps

How to run:
```
python 3_preprocess_target_tweets.py
```

## Step 4 - Get Training Tweets

This script uses labeled tweets that will serve as training tweets to create a machine learning classifier. Here we utilize labeled datasets from online repositories. Such labeled datasets have been labeled by human annotators for positive, negative, and neutral sentiment class. Note that the Twitter terms of service do not permit direct distribution of tweet content and so tweet IDs (references to the original tweets), with their respective sentiment labels, are often made available without the original tweet text and associated meta-data. These datasets can be found in the folder 'files/training_tweets'. As a consequence, we will have to use the Twitter API to retrieve the full tweet content, the tweet text, and the meta-data, by searching for the tweet ID. Some tweets will appear not to be available from the Twitter API and this, in some cases, results in the training datasets having fewer tweets than originally included in the published datasets.

We provide tweets IDs and labeled for the following datasets:

![ScreenShot](/files/readme/training_tweets.png)

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
```
python 4_get_training_tweets.py
```


## Step 5 - Preprocess the Training Tweets

This script is similar to the script in step 3, only that it performs preprocessing on the training tweets. Both the target tweets and training tweets are preprocessed similarly. However, the target tweets here are not filtered for non-English text (we already know they are all English), are not filtered for retweets, and are not filtered for occupation.

How to run:
```
python 5_preprocess_training_tweets.py
```

## Step 6 - Train Several Machine Learning Classifiers

This script will train various machine learning classifiers to predict positive, negative, and neutral sentiment classes on the training tweets. The created ML models can then be used to predict sentiment classes on the target tweets. The script can easily be adjusted to allow for other machine learning classifiers and parameter/hyper-parameter values for grid-search. In the paper we have explored the following models and hyper-parameter values.

![ScreenShot](/files/readme/ml_classifiers.png)

How to run:
```
python 6_train_ml_classifier.py
```


## Step 7 - Infer Sentiment Class of Target Tweets
This script will use one of the created machine learning classifiers from step 6 and infer the sentiment class label for the target tweets. Any created model can be used, but preferably the model that has the highest F1 score should be used. It will infer a positive, negative, or neutral class label for the target tweets. In addition, it creates a labels.pkl file that contains a matrix with tweet_id, label, and tweet_type. Coded as follows:

label : negative = 0, neutral = 1, positive = 2
tweet_type : 0 = interdisciplinary, 1 = transdisciplinary, 2 = multidisciplinary


How to run:
```
python 7_classify_target_tweets.py

```

## Step 8 - Plot Results

Create various plots
- Donutplot showing sentiments per mode of research (int./trans./mult.)
- Sentiment over time shown as a stacked bar chart per week
- Sentiment by occuptation, only showing the most positive occupations
- Bar chart showing the frequency of user tags, @, and URLs for each sentiment and each mode of research

There are 4 switches that can be turned on or off (by setting their value to True or False). Each switch will create the corresponding plot.

*	create_donot_plot = [True|False]
*	create_time_stacked_bar_plot = [True|False]
*	create_sentiment_by_occupation = [True|False]
*	create_twitter_tokens_bar_plot = [True|False]

How to run:
```
python 8_plot_results.py
```


# Description of the training datasets

## Sanders Dataset

The Sanders dataset consists out of 5,513 hand classified tweets related to the topics Apple (@Apple), Google (#Google), Microsoft (#Microsoft), and Twitter (#Twitter). Tweets were classified as positive, neutral, negative, or irrelevant; the latter referring to non-English tweets which we discarded. The Sanders dataset has been used for boosting Twitter sentiment classification using different sentiment dimensions, combining automatically and hand-labeled twitter sentiment labels, and combining community detection and sentiment analysis. The dataset is available from http://www.sananalytics.com/lab/

## SemEval 2016 Dataset

The Semantic Analysis in Twitter Task 2016 dataset, also known as SemEval-2016 Task 4, was created for various sentiment classification tasks. The tasks can be seen as challenges where teams can compete amongst a number of sub-tasks, such as classifying tweets into positive, negative and neutral sentiment, or estimating distributions of sentiment classes. Typically, teams with better classification accuracy or other performance measure rank higher. The dataset consist of training, development, and development-test data that combined consist of 3,918 positive, 2,736 neutral, and 1,208 negative tweets. The original dataset contained a total of 10,000 tweets -- 100 tweets from 100 topics. Each tweet was labeled by 5 human annotators and only tweets for which 3 out of 5 annotators agreed on their sentiment label were considered. The dataset is available from http://alt.qcri.org/semeval2016/task4/

## CLARIN-13 Dataset

The CLARIN 13-languages dataset contains a total of 1.6 million labeled tweets from 13 different languages, the largest sentiment corpus made publicly available. We used the English subset of the dataset since we restricted our analysis to English tweets. Tweets were collected in September 2013 by using the Twitter Streaming API to obtain a random sample of 1% of all publicly available tweets. The tweets were manually annotated by assigning a positive, neutral, or negative label by a total of 9 annotators; some tweets were labeled by more than 1 annotator or twice by the same annotator. For tweets with multiple annotations, only those with two-third agreement were kept. The original English dataset contained around 90,000 labeled tweets. After recollection, a total of 15,064 positive, 24,263 neutral, and 12,936 negative tweets were obtained. The dataset is available from http://hdl.handle.net/11356/1054

## HCR Dataset

The Health Care Reform (HCR) dataset was created in 2010 -- around the time the health care bill was signed in the United States -- by extracting tweets with the hashtag #hcr. The tweets were manually annotated by the authors by assigning the labels positive, negative, neutral, unsure, or irrelevant. The dataset was split into training, development and test data. We combined the three different datasets that contained a total of 537 positive, 337 neutral, and 886 negative tweets. The tweets labeled as irrelevant or unsure were not included. The HCR dataset was used to improve sentiment analysis by adding semantic features to tweets. The dataset is available from https://bitbucket.org/speriosu/updown

## OMD Dataset

The Obama-McCain Debate (OMD) dataset contains 3,238 tweets collected in September 2008 during the United States presidential debates between Barack Obama and John McCain. The tweets were collected by querying the Twitter API for the hash tags #tweetdebate, #current, and #debate08. A minimum of three independent annotators rated the tweets as positive, negative, mixed, or other. Mixed tweets captured both negative and positive components. Other tweets contained non-evaluative statements or questions. We only included the positive and negative tweets with at least two-thirds agreement between annotators ratings; mixed and other tweets were discarded. The OMD dataset has been used for sentiment classification by social relations, polarity classification, and sentiment classification utilizing semantic concept features. The dataset is available from https://bitbucket.org/speriosu/updown

Rating Codes: 1 = negative, 2 = positive, 3 = mixed, 4 = other

## Stanford Test Dataset

The Stanford Test dataset contains 182 positive, 139 neutral, and 177 negative annotated tweets. The tweets were labeled by a human annotator and were retrieved by querying the Twitter search API with randomly chosen queries related to consumer products, company names and people. The Stanford Training dataset, in contrast to the Stanford Test dataset, contains 1.6 million labeled tweets. However, the 1.6 million tweets were automatically labeled, thus without a human annotator, by looking at the presence of emoticons. For example, tweets that contained the positive emoticon :-) would be assigned a positive label, regardless of the remaining content of the tweet. Similarly, tweets that contained the negative emoticon :-( would be assigned a negative label. Such an approach is highly biased and we choose not to include this dataset for the purpose of creating a sentiment classifier from labeled tweets. The Stanford Test dataset, although relatively small, has been used to analyze and represent the semantic content of a sentence for purposes of classification or generation, semantic smoothing to alleviate data sparseness 
problem for sentiment analysis, and sentiment detection of biased and noisy tweets. The dataset is available from http://www.sentiment140.com/

first column provides the sentiment label
'0' = negative
'2'	= neutral
'4' = positive

## Manual Dataset

A random subset of target tweets (tweets referring to the three modes of research) that were manually labeled. Adding a subset of target tweets to the training data allows for more accurate classification predictions since the words and phrases found within the target tweets might not necessarily exist in the training tweets that we used from various online repositories.
