from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import csv
import os
import matplotlib.pyplot as plt
import numpy as np



class Tweet:

    sentimentname = ''
    sentimentvalue = 0

    def __init__(self, user_name, date, text):
        self.user_name = user_name
        self.date = date
        self.text = text



def generateTweetList(path, TweetList):

    # user_name, user_location, user_description, user_created, user_followers, user_friends, user_favourites, user_verified, date, text, hashtags, source, is_retweet
    # only name, date and text of the tweet is important. Therefore, we're only taking them from our datasource
    with open(path, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
    
        for row in csv_reader:
            tw = Tweet(row[0], row[8], row[9])
            TweetList.append(tw)

    return TweetList



def correctTweetList(TweetList):

    text = []

    # replace '\n' with whitespace character to prevent any corruption caused because of them
    for tw in TweetList:
        text = tw.text.split('\n')
        tw.text = " ".join(text)

    # replace 'mention', 'links' and 'tags' in order to prevent any corruption in data caused because of them
    for tw in TweetList:
        tweet_words = []

        for word in tw.text.split(' '):
            if word.startswith('@') and len(word) > 1:
                word = '@user'
            elif word.startswith('http'):
                word = "http"
            elif word.startswith('#'):
                word = '#Tag'
        
            tweet_words.append(word)

        tw.text = " ".join(tweet_words)
    
    return TweetList


def analyzeTweet(TweetList):
    # load model and tokenizer
    # https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment

    roberta = "cardiffnlp/twitter-roberta-base-sentiment"

    model = AutoModelForSequenceClassification.from_pretrained(roberta)

    tokenizer = AutoTokenizer.from_pretrained(roberta)

    labels = ['Negative', 'Neutral', 'Positive']

    # Sentiment analysis
    for tw in TweetList:
        encoded_tweet = tokenizer(tw.text,return_tensors='pt')

        # output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
        output = model(**encoded_tweet)

        # get scores inside an array and get scores to values between 0 and 1
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        # get scores for each label and choose which is the highest
        # Negative 0.95884514  
        # Neutral 0.03684877   
        # Positive 0.0043061352
        for i in range(len(scores)):
            l = labels[i]
            s = scores[i]
            if s > tw.sentimentvalue:
                tw.sentimentvalue = s
                tw.sentimentname = l

    return TweetList


def writeTweetList(path, TweetList):

    neg = 0
    pos = 0
    net = 0

    # All the valued data is stored again inside a csv file.
    with open(path, mode='w',  encoding="utf-8") as ValuedTweet_File:
        tweet_writer = csv.writer(ValuedTweet_File, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for tw in TweetList:
            tweet_writer.writerow([tw.user_name, tw.date, tw.text, tw.sentimentname, tw.sentimentvalue])
            if tw.sentimentname == 'Positive':
                pos += 1
            elif tw.sentimentname == 'Neutral':
                net += 1
            else:
                neg += 1
    
    chart = np.array([pos, net, neg])
    mylabels = ["Positive", "Neutral", "Negative"]

    plt.pie(chart, labels= mylabels, explode=(0, 0, 0.3), autopct='%1.1f%%')
    plt.title("Total Tweets Covered = " + str(len(TweetList)))
    plt.show()