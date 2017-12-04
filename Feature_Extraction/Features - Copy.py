import NRC_Lexicon
import Bing_Liu_Lexicon
import MPQA_Sub_Lexicon
import Word_Clusters
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer
#import nltk

class Features():

	def __init__(self):
		pass

	def construct_features(self,tokenized_tweet,nrc_lexicons,bing_liu,mpqa,clusters, negations):
		#print "Tweet : ",tokenized_tweet
		f=[]
		#NRC Lexicon
		#tokenized_tweet=['hello','world','great','worst']
		
		#[min, max, avg of lexicon]
		#print nrc_lexicons.get_features(tokenized_tweet)
		f += nrc_lexicons.get_features(tokenized_tweet)

		#Bing_Liu Lexicon
		
		#[no_of_positive_words, no_of_negative_words]
		#print bing_liu.get_features(tokenized_tweet)
		#f += bing_liu.get_features(tokenized_tweet)

		#MPQA_SUb_Lexicon
		
		#print mpqa.get_features(tokenized_tweet)
		f += mpqa.get_features(tokenized_tweet)
		#print f
		
		#Find 1000 clusters
		#f += clusters.get_features(tokenized_tweet)
		
		#Negation words
		f += negations.get_features(tokenized_tweet)
		
		
		from twitterTokenizer import Tokenizer
		tokenizer = Tokenizer()
		
		#Char Grams
		char_gram = HashingVectorizer(strip_accents='unicode', binary=True, ngram_range=(4,5), stop_words=None, lowercase=True, analyzer='char', tokenizer=tokenizer.tokenize, n_features=22000)
		char_gram_features = char_gram.fit_transform([' '.join(tokenized_tweet)])
		char_grams = char_gram_features.toarray()
		print len(char_grams[0])
		#print len(f)
		return f
		
	def extract_tweets(self,file,nrc_lexicons,bing_liu,mpqa, clusters, negations):
		fp_train = open(file)
		data = fp_train.read().split('\n')
		print "Number of tweets : ",len(data)
		tweets = {}
		tweet_ids = []
		X=[]
		Y=[]
		for tweet_content in data:
			tweet_content = tweet_content.strip()
			#tweets[str(tweet_content.split('\t')[0])] = [tweet_content[-1],tweet_content[-2]]
			tweet = tweet_content.split('\t')
			if len(tweet)<3:
				continue
			try:
				tweet_features = self.construct_features(tweet[-1].split(),nrc_lexicons,bing_liu,mpqa, clusters, negations)
				X.append(tweet_features)
				if tweet[-2].strip() == 'positive':
					y = 1
				elif tweet[-2].strip() == 'negative':
					y = -1
				elif tweet[-2].strip() == 'neutral':
					y = 0
				else:
					y = int(str(tweet[-2]).strip())
				Y.append(y);
				tweets[str(tweet_content.split('\t')[0])] = [tweet_content.split('\t')[-1],tweet_content.split('\t')[-2]]
				tweet_ids.append(str(tweet_content.split('\t')[0]))
			except Exception as e:
				print e
				print tweet_content.split('\t')
				print tweet_content
				del X[-1]
				
		return [tweets, tweet_ids, X, Y]
	
if __name__ == '__main__':
	train_file = '../Data/twitter-2016train-A.txt'
	[X_train, Y_train] = extract_tweets(train_file)
	print len(X_train)
	print len(Y_train)
	
	
	test_file = '../Data/twitter-2016test-A.txt'
	[X_test, Y_test] = extract_tweets(test_file)
	print len(X_test)
	print len(Y_test)