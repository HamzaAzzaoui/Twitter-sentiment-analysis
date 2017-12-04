
class NRC_Lexicon():
	def __init__(self):
		self.file1 = 'Feature_Extraction/Resources/HS-AFFLEX-NEGLEX-unigrams.txt'
		self.file2 = 'Feature_Extraction/Resources/HS-AFFLEX-NEGLEX-bigrams.txt'
		self.d1 = {}
		self.d2 = {}
	def construct_lexicons(self):
		self.fp = open(self.file1)
		data = self.fp.read().split('\n')
		# great 10.32 2318 0  (Each line content)
		self.fp.close()
		for line in data:
			try:
				line_content = line.split()
				self.d1[line_content[0].strip()] = float(line_content[1].strip())
			except Exception as e:
				print e
				pass
		## Bigram Lexicons
		self.fp = open(self.file2)
		data = self.fp.read().split('\n')
		#. #great	8.988	6213	0  (Each line content)
		self.fp.close()
		for line in data:
			try:
				line_content = line.split()
				self.d2[' '.join(line_content[0:2])] = float(line_content[2].strip())
			except Exception as e:
				print e, 'bigram'
				pass
		return 0
		
	def get_features(self,tokenized_tweet):
		min_lexicon=0
		max_lexicon=0
		avg_lexicon=0
		polarity_count = 0
		for word in tokenized_tweet:
			try:
				if min_lexicon > self.d1[word]:
					min_lexicon = self.d1[word]
				if max_lexicon < self.d1[word]:
					max_lexicon = self.d1[word]
				polarity_count+=float(self.d1[word])
			except Exception as e:
				#print e, 'exception unigram'
				pass
		avg_lexicon=polarity_count*(1.0)/len(tokenized_tweet)
		
		min_lexicon_bi=0
		max_lexicon_bi=0
		avg_lexicon_bi=0
		polarity_count_bi = 0
		for i in range(len(tokenized_tweet)-1):
			bigram = ' '.join(tokenized_tweet[i:i+2])
			#print bigram
			try:
				if min_lexicon_bi > self.d2[bigram]:
					min_lexicon_bi = self.d2[bigram]
				if max_lexicon_bi < self.d2[bigram]:
					max_lexicon_bi = self.d2[bigram]
				polarity_count_bi += float(self.d2[bigram])
			except Exception as e:
				#print e, 'Exception bigram'
				pass
		avg_lexicon_bi = polarity_count_bi*(1.0)/len(tokenized_tweet)
		#print [min_lexicon,max_lexicon,avg_lexicon]
		return [min_lexicon,max_lexicon,avg_lexicon] + [min_lexicon_bi, max_lexicon_bi, avg_lexicon_bi]
		
if __name__ == "__main__":
	nrc = NRC_Lexicon()
	nrc.construct_lexicons()
	#print nrc.d2
	tokenized_tweet = ['.', '#great', 'teacher']
	print nrc.get_features(tokenized_tweet)