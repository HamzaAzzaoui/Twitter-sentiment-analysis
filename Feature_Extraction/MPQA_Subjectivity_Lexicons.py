
# coding: utf-8

# In[7]:

fp_mpqa = file('Resources/subjclueslen1-HLTEMNLP05.tff')
data=fp_mpqa.read()
mpqa_lexicon = {}
mpqa_p_count=0
mpqa_n_count=0
mpqa_neutral_count=0
mpqa_both_count=0


# In[8]:

all_words = data.split('\n')


# In[9]:

for word_content in all_words:
    word_content_list = word_content.split()
    try:
    #print word_content_list[2].split('=')
        mpqa_lexicon[word_content_list[2].split('=')[1]]={'subjectivity':word_content_list[0].split('=')[1],'polarity':word_content_list[-1].split('=')[1]}
    except Exception as e:
        print "EXceptin"
        print word_content,e
        #print word_content_list[2].split('=')[1]
        #print word_content_list[1].split('=')[1]


# In[10]:

print word_content_list
print mpqa_lexicon['personages']


# In[11]:

tokenized_tweet=['hello','world','great','worst']
for word in tokenized_tweet:
    try:
        print mpqa_lexicon[word]
        if mpqa_lexicon[word]['subjectivity']=='weaksubj':
            subject=1
        else:
            subject=2
        if mpqa_lexicon[word]['polarity']=='positive':
            mpqa_p_count = mpqa_p_count+1*subject
        elif mpqa_lexicon[word]['polarity']=='negative':
            mpqa_n_count = mpqa_n_count+1*subject
        elif mpqa_lexicon[word]['polarity']=='both':
            mpqa_both_count = mpqa_both_count+1*subject
        elif mpqa_lexicon[word]['polarity']=='neutral':
            mpqa_neutral_count = mpqa_neutral_count+1*subject
    except Exception as e:
        print e


# In[12]:

print mpqa_p_count
print mpqa_n_count
print mpqa_neutral_count
print mpqa_both_count


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



