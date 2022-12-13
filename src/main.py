from util import *


TweetList = []

path = os.getcwd()

#path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

TweetList = generateTweetList(path + '\\data\\NFT_tweets.csv',TweetList)

TweetList = correctTweetList(TweetList)

TweetList = analyzeTweet(TweetList)

writeTweetList(path + '\\data\\Tweets_Valued.csv', TweetList)

