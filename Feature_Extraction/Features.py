import Main_Features

def FeatureExtraction(tweet_text_list, features, nrc_lexicons, bing_liu, mpqa, negations):
	index=0
	for tweet in tweet_text_list:
        #ngrams = feature_ngrams(tweet)
        #charGrams = feature_char_grams(tweet)
		exclamtionCount = Main_Features.feature_excalamation_count(tweet)
		questionCount = Main_Features.feature_question_count(tweet)
		Exclamation_Question_Count = Main_Features.feature_excalamation_and_question_count(tweet)
		ElongatedCount= Main_Features.feature_elongated_count(tweet)
		Positive_emoticonCount = Main_Features.feature_positive_emoticon_count(tweet)
		Negative_emoticonCount = Main_Features.feature_negative_emoticon_count(tweet)
		emoticon_existsBinary = Main_Features.feature_emoticon_exists_binary(tweet)
		semantic_lexicons = feature_semantic(tweet, nrc_lexicons, bing_liu, mpqa, negations)
		features.loc[index]=(tweet,exclamtionCount,questionCount,Exclamation_Question_Count,ElongatedCount,0,
						Positive_emoticonCount,Negative_emoticonCount,emoticon_existsBinary)+tuple(semantic_lexicons)
		index+=1
	ngrams_train = Main_Features.feature_ngrams(tweet_text_list)
	charGrams_train = Main_Features.feature_char_grams(tweet_text_list)
    #features['n-grams'] = ngrams
    #features['char-grams']=charGrams
	return features,ngrams_train,charGrams_train
	
def FeatureExtraction_test(tweet_text_list, features, nrc_lexicons, bing_liu, mpqa, negations):
	index=0
	for tweet in tweet_text_list:
        #ngrams = feature_ngrams(tweet)
        #charGrams = feature_char_grams(tweet)
		exclamtionCount = Main_Features.feature_excalamation_count(tweet)
		questionCount = Main_Features.feature_question_count(tweet)
		Exclamation_Question_Count = Main_Features.feature_excalamation_and_question_count(tweet)
		ElongatedCount = Main_Features.feature_elongated_count(tweet)
		Positive_emoticonCount = Main_Features.feature_positive_emoticon_count(tweet)
		Negative_emoticonCount = Main_Features.feature_negative_emoticon_count(tweet)
		emoticon_existsBinary = Main_Features.feature_emoticon_exists_binary(tweet)
		semantic_lexicons = feature_semantic(tweet, nrc_lexicons, bing_liu, mpqa, negations)
		features.loc[index]=(tweet,exclamtionCount,questionCount,Exclamation_Question_Count,ElongatedCount,0,
                            Positive_emoticonCount,Negative_emoticonCount,emoticon_existsBinary)+tuple(semantic_lexicons)
		index+=1
	return features
	
def FeatureExtraction_final_test(tweet_text_list, features, nrc_lexicons, bing_liu, mpqa, negations):
	index=0
	for tweet in tweet_text_list:
        #ngrams = feature_ngrams(tweet)
        #charGrams = feature_char_grams(tweet)
		exclamtionCount = Main_Features.feature_excalamation_count(tweet)
		questionCount = Main_Features.feature_question_count(tweet)
		Exclamation_Question_Count = Main_Features.feature_excalamation_and_question_count(tweet)
		ElongatedCount = Main_Features.feature_elongated_count(tweet)
		Positive_emoticonCount = Main_Features.feature_positive_emoticon_count(tweet)
		Negative_emoticonCount = Main_Features.feature_negative_emoticon_count(tweet)
		emoticon_existsBinary = Main_Features.feature_emoticon_exists_binary(tweet)
		semantic_lexicons = feature_semantic(tweet, nrc_lexicons, bing_liu, mpqa, negations)
		features.loc[index]=(tweet,exclamtionCount,questionCount,Exclamation_Question_Count,ElongatedCount,0,
                            Positive_emoticonCount,Negative_emoticonCount,emoticon_existsBinary)+tuple(semantic_lexicons)
		index+=1
	return features
	
	
def feature_semantic(tweet, nrc_lexicons, bing_liu, mpqa, negations):
	f = []
	tokenized_tweet = tweet.split()
	#NRC Lexicons Uni and Bigrams
	f += nrc_lexicons.get_features(tokenized_tweet)
	
	#Bing_Liu Lexicon	
	#[no_of_positive_words, no_of_negative_words]
	f += bing_liu.get_features(tokenized_tweet)

	#MPQA_SUb_Lexicon
	
	#print mpqa.get_features(tokenized_tweet)
	f += mpqa.get_features(tokenized_tweet)
	#print f
	
	#Find 1000 clusters
	#f += clusters.get_features(tokenized_tweet)
	
	#Negation words
	f += negations.get_features(tokenized_tweet)
	
	return f