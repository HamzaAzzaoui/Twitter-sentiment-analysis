from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer
import re
#from Feature_Extraction import twitterTokenizer
import twitterTokenizer
tokenizer = twitterTokenizer.Tokenizer()

def feature_ngrams(tweet_text):   #method to get ngrams (n = [1,2,3,4,..]) of a tweet in taining set
    ngram = HashingVectorizer(strip_accents='unicode', binary=True, ngram_range=(1,4), stop_words=None, lowercase=True,  tokenizer=tokenizer.tokenize, n_features=10000)
    ngram_features = ngram.fit_transform(tweet_text)
    return ngram_features
    #return list(nltk.ngrams(tweet_text,n))
def feature_ngrams_test(tweet_text):
    ngram = HashingVectorizer(strip_accents='unicode', binary=True, ngram_range=(1,4), stop_words=None, lowercase=True,  tokenizer=tokenizer.tokenize, n_features=10000)
    ngram_features = ngram.fit_transform(tweet_text)
    return ngram_features
def feature_char_grams(tweet_text):   #method to get ngrams (n = [1,2,3,4,..]) of a tweet in a training set
    char_gram = HashingVectorizer(strip_accents='unicode', binary=True, ngram_range=(4,5), stop_words=None, lowercase=True, analyzer='char', tokenizer=tokenizer.tokenize, n_features=22000)
    char_gram_features = char_gram.fit_transform(tweet_text)
    return char_gram_features
def feature_charGrams_test(tweet_text):
    char_gram = HashingVectorizer(strip_accents='unicode', binary=True, ngram_range=(4,5), stop_words=None, lowercase=True, analyzer='char', tokenizer=tokenizer.tokenize, n_features=22000)
    char_gram_features = char_gram.fit_transform(tweet_text)
    return char_gram_features
def feature_excalamation_count(tweet_text): #method to get frequency of exclamation(!) in a tweet
    return tweet_text.count("!")
    
def feature_question_count(tweet_text):  #method to get frequency of question(?) in a tweet
    return tweet_text.count("?")
    
def feature_excalamation_and_question_count(tweet_text):#method to get frequency of question mark(?) & exclamation (!) in a tweet
    return tweet_text.count("?") + tweet_text.count("!")
    
def feature_elongated_count(tweet_text): #method to get frequency of elongated words and capitalized words
    count_capitalized=0
    count_elongated=0
    for i in tweet_text.split():
        if i.isupper():
            count_capitalized+=1
        pattern = re.compile(r'(.)\1?')
        result = [x.group() for x in pattern.finditer(i)]
        filtered_result = [x for x in result if len(x) == 2]
        if len(filtered_result) > 2:
            count_elongated+=1
        return count_capitalized+count_elongated

def feature_positive_emoticon_count(tweet_text): #method to get frequency of positive emoticons in a tweet
    emoticons_pos = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # nose
      [\)\]dD\}@]                # mouth      
      |                          # reverse order now! 
      [\)\]dD\}@]                # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""
    capture_pos_emoticon=re.compile(emoticons_pos,re.VERBOSE|re.I|re.UNICODE)
    capture_pos_emoticon_list=capture_pos_emoticon.findall(tweet_text)
    return len(capture_pos_emoticon_list)
    
def feature_negative_emoticon_count(tweet_text): #method to get frequency of negative emoticons in a tweet
    emoticons_neg = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\(\[pP/\:\{\|] # mouth      
      |                          # reverse order now! 
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
      )"""
    capture_neg_emoticon=re.compile(emoticons_neg,re.VERBOSE|re.I|re.UNICODE)
    capture_neg_emoticon_list=capture_neg_emoticon.findall(tweet_text)
    return len(capture_neg_emoticon_list)
#method which checks the existence of emoticon in a tweet and returns a binary
def feature_emoticon_exists_binary(tweet_text):
    if feature_negative_emoticon_count(tweet_text) or feature_positive_emoticon_count(tweet_text):
        return 1
    else:
        return 0