
class MPQA_Sub_Lexicon():
	
	def __init__(self):
		self.fp_mpqa = file('Feature_Extraction/Resources/subjclueslen1-HLTEMNLP05.tff')
		self.d = {}
	def construct_lexicons(self):
		data = self.fp_mpqa.read()
		all_words = data.split('\n')

		for word_content in all_words:
			#type=weaksubj len=1 word1=abandoned pos1=adj stemmed1=n priorpolarity=negative
			word_content_list = word_content.split()
			try:
				polarity =  word_content_list[-1].split('=')[1]
				subjectivity = word_content_list[0].split('=')[1]
				word = word_content_list[2].split('=')[1]
				if polarity == 'negative':
					if subjectivity == 'strongsubj':
						self.d[word] = -2
					elif subjectivity == 'weaksubj':
						self.d[word] = -1
						
				elif polarity == 'positive':
					if subjectivity == 'strongsubj':
						self.d[word] = 2
					elif subjectivity == 'weaksubj':
						self.d[word] = 1
				
				elif polarity == 'neutral' or polarity == 'both':
					if subjectivity == 'strongsubj':
						self.d[word] = 0
					elif subjectivity == 'weaksubj':
						self.d[word] = 0
						
			except:
				print word_content
				pass
		return self.d

	def get_features(self, tokenized_tweet):
		mpqa_p_count=0
		mpqa_n_count=0
		mpqa_neutral_count=0
		for word in tokenized_tweet:
			try:
				if self.d[word]>0:
					mpqa_p_count+=self.d[word]
				elif self.d[word]<0:
					mpqa_n_count+=self.d[word]
				else:
					mpqa_neutral_count+=self.d[word]
			except Exception as e:
				#print e
				pass
		return [mpqa_p_count,mpqa_n_count,mpqa_neutral_count]
		#return [mpqa_p_count,mpqa_n_count]
		
if __name__ == "__main__":
	mpqa = MPQA_Sub_Lexicon()
	mpqa_d = mpqa.construct_lexicon()
	print len(mpqa_d.keys())
	print mpqa_d['yeah']