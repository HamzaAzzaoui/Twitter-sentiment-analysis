
# coding: utf-8

# In[1]:

fp_positive=open('positive-words.txt')
fp_negative=open('negative-words.txt')


# In[2]:

positive_words=fp_positive.read()
negative_words=fp_negative.read()


# In[3]:

positive_word_count = 0
tokenized_tweet=['hello','world']
for word in tokenized_tweet:
    if word in positive_words:
        positive_word_count+=1
    else if word in negative_words:
        n
print positive_word_count


# In[ ]:




# In[ ]:



