import nltk

tokenized_tweet = ['i', 'am', 'good']

features = []

#POS TAGS counting
#of adjectives
#of comparative adjectives
#of superlative adjectives
tagged_words = nltk.pos_tag(tokenized_tweet)
pos_tags = [word[1] for word in tagged_words]
pos_features=[pos_tags.count('JJ'),pos_tags.count('JJR'),pos_tags.count('JJS')]
features+=pos_features;

print features

