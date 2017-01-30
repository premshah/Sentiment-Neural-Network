
# coding: utf-8

# In[ ]:

#import nltk
#nltk.download()


# In[46]:

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter


# In[47]:

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000


# In[48]:

def create_lexicon(pos,neg):
    '''Open all files in read mode. Read all lines of each file as content, then read all words of each line
    of content as word-tokens and add it to list of words.'''
    lexicon = []
    for fi in [pos,neg]:
        with open(fi,'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower().decode('utf8') )
                lexicon += list(all_words)
                
    '''lematize all words from list of words and get count of occurence of each word.
    lematize meaning run and running is have same root word. so it is converting all words to their root words'''
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    '''w_counts will be of format {'and' : 400000, 'the' : 34560}'''
    w_counts = Counter(lexicon)
    
    '''we only want that words in our lexicon which are not so frequent and also which are not very rare thats why,
    we choose words having occurence between 1000 and 50'''
    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    print(len(l2))
    return l2


# In[49]:

def sample_handling(sample,lexicon,classification):
    '''initialize featureset to empty.
    open sample in read mode, read all words from all lines of sample and lemmatize them.
    count features of all words of each line.
    add features of words of each line as a list in featureset.
    thus featureset will be a list of list of features.
    each list inside featureset will be of size of lexicon'''
    featureset = []
    with open(sample,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower().decode('utf8') )
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower().decode('utf8') )
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])
            
    return featureset


# In[54]:

def create_feature_sets_and_labels(pos,neg,test_size=0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling(pos,lexicon,[1,0])
    features += sample_handling(neg,lexicon,[0,1])
    random.shuffle(features)
    
    features = np.array(features)
    testing_size = int(test_size*len(features))
    
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])
    
    return train_x, train_y, test_x, test_y


# In[ ]:

if __name__ == '__main__':
    train_x, train_y, test_x, test_y =  create_feature_sets_and_labels('/home/prem/Desktop/Projects/Word Neural Network/pos.txt','/home/prem/Desktop/Projects/Word Neural Network/neg.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)


# In[ ]:



