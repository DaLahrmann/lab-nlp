# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:52:51 2022

@author: lahrm
"""
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import nltk

tag_dict = {"J": wordnet.ADJ, 
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper() 
    # gets first letter of POS categorization
    return tag_dict.get(tag, wordnet.NOUN) # get returns second argument if first key does not exist 

def clean_up(s):
    """
    Cleans up numbers, URLs, and special characters from a string.

    Args:
        s: The string to be cleaned up.

    Returns:
        A string that has been cleaned up.
    """
    s=re.sub('http\S+','',s) #removing urls
    s=re.sub('\W',' ',s) #replace all special signs with a white space
    s=re.sub('\d',' ',s) #replace all numbers with a white space
    return s.lower()

def tokenize(s):
    """
    Tokenize a string.

    Args:
        s: String to be tokenized.

    Returns:
        A list of words as the result of tokenization.
    """
    return word_tokenize(s)


def stem_and_lemmatize(l):
    """
    Perform stemming and lemmatization on a list of words.

    Args:
        l: A list of strings.

    Returns:
        A list of strings after being stemmed and lemmatized.
    """
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer() 
    
    l1=[ps.stem(w) for w in l]
    l2=[lemmatizer.lemmatize(w,get_wordnet_pos(w)) for w in l] 
    # lemitizer also use wordtype for better solutions
    return l1, l2


def remove_stopwords(l):
    """
    Remove English stopwords from a list of strings.

    Args:
        l: A list of strings.

    Returns:
        A list of strings after stop words are removed.
    """
    without_sw = [word for word in l if not word in stopwords.words()]
    without_sw2= [word for word in without_sw if len(word)>1]
    # also removes single letters
    return without_sw2

def clean_all(x):
    """
    Run all clean functions for a given text
    use the lemmatized list for the final step


    Parameters
    ----------
    x : string (tweet)
    

    Returns
    -------
    lemmatized list of words without stopwords

    """
    x=clean_up(x)
    l=tokenize(x)
    l1,l2= stem_and_lemmatize(l)
    return remove_stopwords(l2)

def find_feature(row,wordlist):
    words=set(row.text_processed)
    features={}
    for w in wordlist:
        features[w] = w in words
    return features, row.target == 4

def categorization_maker(data,n=5000):
    """
    

    Parameters
    ----------
    data : Dataframe of tweet collection.
    n : size of  word_bag, optional
        The default is 5000.
        

    Returns
    -------
    categorisation, and accuracy of categorization

    """
    data['text_processed']=data.text.apply(clean_all)
    words=data.text_processed.sum()
    fdist = nltk.FreqDist(words)
    wordlist1=fdist.most_common(5000)
    wordlist=[x[0] for x in wordlist1]
    feature=list(data.apply(lambda x:find_feature(x,wordlist),axis=1))
    m=len(feature)//5
    test=feature[:4000] #train test split (len(test)=0.2 len(train))
    train=feature[4000:]
    print(train[:5])
    classifier=nltk.NaiveBayesClassifier.train(train)
    return classifier,nltk.classify.accuracy(classifier,test)
    
