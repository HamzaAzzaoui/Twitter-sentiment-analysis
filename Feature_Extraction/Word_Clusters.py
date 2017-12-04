class Word_Clusters():
	def __init__(self):
		self.d={}
		self.values=[]
		self.file='Feature_Extraction/Resources/50mpaths2.txt'
	def construct_clusters(self):
		self.fp=open(self.file)
		self.data=self.fp.read()
		for line in self.data.split('\n'):
			cluster=line.split()[0]
			word=line.split()[1]
			self.d[word]=cluster
		#print self.d
		self.fp.close()
		self.values = list((set(self.d.values())))
		return self.d

	def get_features(self, tokenized_tweet):
		r = [0]*(len(self.values))
		#print len(r)
		for word in tokenized_tweet:
			if self.d.has_key(word):
				index = self.values.index(self.d[word])
				#print index
				if r[index]==0:
					r[index] = 1
				else:
					r[index]+=1
		#print len(r)
		#print 'number of clusters',r.count(1)
		return r

if __name__ == "__main__":
    clust = Word_Clusters()
    clust_d= clust.construct_clusters()
    values = clust_d.values()
    print len(set(values))


# In[ ]:




# In[ ]:



