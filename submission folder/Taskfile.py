#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Library Imports
import numpy as np
import pandas as pd
import nltk
import os


# In[4]:


#XLSX file to pandas dataframe

data = pd.read_excel('./Input.xlsx')
df = pd.DataFrame(data)


# CREATION OF A LIST OUT OF THE GIVEN LIST OF POSITIVE AND NEGATIVE
# 
# WORDS GIVEN IN THE MASTERDICTIONARY

# In[5]:


positive = []

with open('./MasterDictionary/positive-words.txt', 'r') as pf:
    line = pf.readlines()
    for lin in line:
        k = lin.replace('\n','')
        positive.append(k)
    


# In[6]:


positive


# In[7]:


negative = []

with open('./MasterDictionary/negative-words.txt', 'r') as pf:
    line = pf.readlines()
    for lin in line:
        k = lin.replace('\n','')
        negative.append(k)


# In[8]:


negative


# In[9]:


main_dir = './StopWords/'
stopw_list = []
for file in os.listdir('./StopWords/'):
    
    with open(os.path.join(main_dir, file), 'rb') as f:
        
        line = f.readlines()
        stopw_list.append(line)


# In[17]:


stopw_list


# In[24]:


stoplist_mod = []
for list in stopw_list:
    
    for i in range(len(list)):
        
        list[i] = repr(list[i])
        
        list[i] = list[i].replace("b", '')
        list[i] = list[i].replace("\r\n", '')
        list[i] = list[i].replace("'", '')
        final = list[i].split('|', 1)[0]
        
        stoplist_mod.append(final)


# In[25]:


for i in range (len(stoplist_mod)):
    
    stoplist_mod[i] = stoplist_mod[i].replace("\\r\\n", "").lower()


# AN IMRPOVISED STOPWORD LIST THAT CONTAINS THE STOPWORDS FROM THE GIVEN FILE
# 
# AS WELL AS NLTK'S STANDARD STOPWORD LIST, TO DO BETTER CLEANING

# In[26]:


stoplist_mod


# SCRAPING AND CREATING A LIST OF PARAGRAPHS

# In[ ]:


import urllib.request

article_list = []

from bs4 import BeautifulSoup as bs 

for i in range(len(df)):
    
    url = df['URL'][i]
    
    try:
        
        html = urllib.request.urlopen(url)
        
        
        html_parse = bs(html, 'html.parser')
        
        para = html_parse.find_all('div',{'class':'td-post-content tagdiv-type'}, 'p')
        
        y = ""
        
        for x in para:
            
            y += x.text
            
        
        y = y.replace('\n','')
        article_list.append(y)
        
    except:
        
        article_list.append('')
        
        print(f"HTTP error at index{i} URL:{df['URL'][i]}")


# In[28]:


for i in range(len(article_list)):
    
    article_list[i] = article_list[i].lower()


# In[29]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words1 = stop_words + stoplist_mod


# In[30]:


from nltk.stem import  WordNetLemmatizer

lem = WordNetLemmatizer()
nltk.download('wordnet')


# In[31]:


def process(paragraph):    # The main cleaning function of the paragraph
                           # to get rid of all unecessary words and punctuations.
        
    word = nltk.word_tokenize(paragraph)
    filtered_para = [lem.lemmatize(w) for w in word if not w.lower() in stop_words1]
    filtered_para_mod = [filt for filt in filtered_para if filt.isalpha()]
    return filtered_para_mod


# In[32]:


print(process(article_list[1])) # Just trying out the main process fuction


# NOTE: THE WORD LIST AFTER TOKENIZING EACH PARAGRAPH IS ALWAYS CLEANED BEFORE DETERMINING
# 
# ANY PARAMETER. 
# 
# SO THE WORD LENGTH OF EACH PARAGRAPH IS DIFFERENT THAN THE ORIGINAL LENGTH

# In[33]:


#These are helper functions to determine the number of positive / negative words

def plus(sent):
    
    countpos = 0    
    
    
    for e in sent:
        
        if e in positive:
            countpos +=1
            
        else:
            countpos+=0
            
    
    return countpos



def minus(sent):
    
    countneg = 0    
    
    for e in sent:
        
        if e in negative:
            countneg +=1
            
        else:
            countneg+=0
            
   
    
    return countneg


# In[34]:


#Functions to obtain the positive score , negative score, Polarity score
#and subjectivity score

def obtainpos(para):
    
    matrixplus = []

    
    
    
    sents = nltk.sent_tokenize(para)
    for sent in sents:
        filtered_words = process(sent)
        
        pos_score = plus(filtered_words)
        matrixplus.append(pos_score)
        
        
    return sum(matrixplus)

def obtainneg(para):
    
    matrixminus = []
    
    
    sents = nltk.sent_tokenize(para)
    for sent in sents:
        filtered_words = process(sent)
        
        neg_score = minus(filtered_words)
        matrixminus.append(neg_score)
        
        
    return sum(matrixminus)


def polarity(para):       #Polarity Score: This is the score that determines if a given text is positive or negative in nature. It is calculated by using the formula: Polarity Score = (Positive Score – Negative Score)/ ((Positive Score + Negative Score) + 0.000001)

    
    return (obtainpos(para)-obtainneg(para))/(obtainpos(para) + obtainneg(para)+0.000001)

def subjectivity(para):
    
    words = process(para)
    
    return(obtainpos(para)+obtainneg(para))/(len(words)+0.000001)


# In[35]:


import re
pronounReg = re.compile(r'\b(i|we|my|ours|(?-i:us))\b', re.I)


# In[36]:


import pyphen
dic = pyphen.Pyphen(lang= 'en')


def numb_syllables(para):          #From What is understood from the task, this function
                                   #returns the list of syllable number per word in a paragraph 
    syllable_list = []
    
    words = process(para)
    for word in words:
        count = 0

        
        list_ = dic.inserted(word).split('-')
        if('es' in list_):
            count+=1
            
        elif('ed' in list_):
            count+=1
            
            
        syllable_list.append(len(list_) - count)
    return(syllable_list)
            
            
        


def num_complexword(para):    #Complex words are words in the text that contain more than two syllables.
    
    
    compword = []
    
    words_mod = process(para)
    
    for word in words_mod:
        
        if(len(dic.inserted(word).split('-'))> 2):
            
            compword.append(word)
            
    return(len(compword))


def percentage_complex(para):
    
   
    words_mod = process(para)
    
    return(num_complexword(para)/len(words_mod))*100


def wordpersent(para):     #Average Number of Words Per Sentence = the total number of words / the total number of sentences

    
    words = 0
    
    sents = nltk.sent_tokenize(para)
    
    for sent in sents:
        wds_mod = process(sent)
        
        words += len(wds_mod)
        
    return words/len(sents)


def avg_sent_len(para):      #Average Sentence Length = the number of words / the number of sentences

    sents = nltk.sent_tokenize(para)
    words = process(para)
    
    return (len(words)/len(sents))


def fog_index(para):         #Fog Index = 0.4 * (Average Sentence Length + Percentage of Complex words)
    
    return 0.4*(avg_sent_len(para)+ percentage_complex(para))

def pronoun_count(para):     #Count the number of personal pronouns in a paragraph
    pronoun_list = pronounReg.findall(para)
    
    return(pronoun_list)

def avg_word_length(para):    #Average Word Length is calculated by the formula:
                             #Sum of the total number of characters in each word/Total number of words

    lentotal = 0
    
    words = process(para)
    for word in words:
        
        lentotal += len(word)
        
    return (lentotal/len(words))

def word_count(para):
    
    words = process(para)
    return len(words)


# In[51]:


dict_list = []


# In[69]:



for i in range(len(article_list)):    #Creation of a list of dictionaries that has all the column names as 
                                    
                                      #as KEYS and the return values of the above functions defined as VALUES
    try:
        dictionary = {}
        
        
        dictionary['POSITIVE SCORE'] = obtainpos(article_list[i])
        dictionary['NEGATIVE SCORE'] = obtainneg(article_list[i])

        dictionary['POLARITY SCORE'] = polarity(article_list[i])
        dictionary['SUBJECTIVITY SCORE'] = subjectivity(article_list[i])
        dictionary['AVG SENTENCE LENGTH'] = avg_sent_len(article_list[i])
        dictionary['PERCENTAGE OF COMPLEX WORDS'] = percentage_complex(article_list[i])
        dictionary['FOG INDEX'] = fog_index(article_list[i])
        dictionary['AVG NUMBER OF WORDS PER SENTENCE'] = wordpersent(article_list[i])
        dictionary['COMPLEX WORD COUNT'] = num_complexword(article_list[i])
        dictionary['WORD COUNT'] = word_count(article_list[i])
        dictionary['SYLLABLE PER WORD'] = numb_syllables(article_list[i])
        dictionary['PERSONAL PRONOUNS'] = pronoun_count(article_list[i])
        dictionary['AVG WORD LENGTH'] = avg_word_length(article_list[i])
        
        
        dict_list.append(dictionary)
        
        
    except:
        
        dictionary = {}
        
        dictionary['POSITIVE SCORE'] = "This para is empty as URL showed error"
        
        dict_list.append(dictionary)
        
        
        


# In[80]:


print(dict_list[103])


# In[76]:


len(dict_list)


# In[83]:


outp_data = pd.read_excel('./Output Data Structure.xlsx')
df2 = pd.DataFrame(outp_data)


# In[84]:


df2


# CONVERTING THE DICTIONARY LIST TO A PANDAS DATAFRAME

# In[88]:


df3 = pd.DataFrame.from_dict(dict_list) 


# In[89]:


df3


# AS THE DATAFRAME DOES NOT HAVE 'URL_ID' AND 'URL' , 
# 
# THOSE FEATURES HAVE BEEN ADDED USING THE df2 DATAFRAME
# 
# DEFINED BY THE 'OUTPUT DATA STRUCTURE.XLSX' FILE

# In[90]:


df3.insert(0, 'URL', df2['URL'])


# In[92]:


df3.insert(0, 'URL_ID', df2['URL_ID'])


# FINALLY CONVERT THE FINAL DATAFRAME TO excel FILE

# In[93]:


df3.to_excel('Output Data Structure_1.xlsx')

