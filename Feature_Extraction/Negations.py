
class Negations():
	def __init__(self):
		self.file = 'Feature_Extraction/Resources/negations.txt'
		self.d = {}
		self.d['negations']=[]
		
	def construct_lexicons(self):
		self.fp = open(self.file)
		data = self.fp.read().split('\n')
		for line in data:
			if len(line)!=0:
				try:
					word = line.strip()
					self.d['negations'].append(word)
				except Exception as e:
					print e
					pass
		return self.d
	
	#returns number of negative words in tweet
	def get_features(self, tokenized_tweet):
		no_of_negations = 0;
		for word in tokenized_tweet:
			if word in self.d['negations']:
				no_of_negations+=1
		return [no_of_negations]
		
		
if __name__ == "__main__":
	neg = Negations()
	neg.construct_lexicons()
	print neg.d
	tweet = ['Im', 'not','good']
	print neg.get_features(tweet);