# Loading Libraries
import nltk
import pandas as pd
import numpy as np
import matplotlib
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import webtext
from nltk.corpus import inaugural
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer 
from nltk.stem import WordNetLemmatizer
from nltk.metrics.distance  import edit_distance
from nltk.corpus import words


#corpus = inaugural.raw(r'C:\Users\shrey\Downloads\shakespeares-works_TXT_FolgerShakespeare\alls-well-that-ends-well_TXT_FolgerShakespeare.txt')
#print(corpus)

#STOPWORD REMOVAL
def stopword_removal(corpus):
    sents = nltk.sent_tokenize(corpus)
    #print("The number of sentences is", len(sents))
    words = nltk.word_tokenize(corpus)
    #print("The number of tokens is", len(words))
    average_tokens = round(len(words)/len(sents))
    #print("The average number of tokens per sentence is",average_tokens)
    unique_tokens = set(words)
    #print("The number of unique tokens are", len(unique_tokens))
    stop_words = set(stopwords.words('english'))
    final_tokens = []
    for each in words:
     if each not in stop_words:
        final_tokens.append(each)
    #print("The number of total tokens after removing stopwords are", len((final_tokens)))
    freq = nltk.FreqDist(final_tokens)
    #freq.plot(20, cumulative=False)
    return final_tokens


#STEMMING
def stemming(final_tokens):
    #Create instances of both stemmers, and stem the words using them.
    stemmer_ps = PorterStemmer()  
    #an instance of Porter Stemmer
    stemmed_words_ps = [stemmer_ps.stem(word) for word in final_tokens]
    #print("Porter stemmed words: ", stemmed_words_ps)
    stemmer_ss = SnowballStemmer("english")   
    #an instance of Snowball Stemmer
    stemmed_words_ss = [stemmer_ss.stem(word) for word in final_tokens]
    #print("Snowball stemmed words: ", stemmed_words_ss)
    return stemmed_words_ps
    # A function which takes a sentence/corpus and gets its stemmed version.
def stemSentence(sentence):
    stemmer_ps = PorterStemmer() 
    token_words=word_tokenize(sentence) #we need to tokenize the sentence or else stemming will return the entire sentence as is.
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(stemmer_ps.stem(word))
        stem_sentence.append(" ") #adding a space so that we can join all the words at the end to form the sentence again.
    return "".join(stem_sentence)
    # stemmed_sentence = stemSentence("The circumstances under which I now meet you will acquit me from entering into that subject further than to refer to the great constitutional charter under which you are assembled, and which, in defining your powers, designates the objects to which your attention is to be given.")
    # print("The Porter stemmed sentence is: ", stemmed_sentence)'

#LEMMATIZATION
def lemmatization(stemmed_words_ps):
    lemmatizer = WordNetLemmatizer()   
    #an instance of Word Net Lemmatizer
    lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words_ps] 
    #print("The lemmatized words: ", lemmatized_words) 
    #prints the lemmatized words
    lemmatized_words_pos = [lemmatizer.lemmatize(word, pos = "v") for word in stemmed_words_ps]
    #print("The lemmatized words using a POS tag: ", lemmatized_words_pos) 
    #prints POS tagged lemmatized words
    return lemmatized_words
    #A function which takes a sentence/corpus and gets its lemmatized version.
def lemmatizeSentence(sentence):
    lemmatizer = WordNetLemmatizer() 
    token_words=word_tokenize(sentence) 
    #we need to tokenize the sentence or else lemmatizing will return the entire sentence as is.
    lemma_sentence=[]
    for word in token_words:
        lemma_sentence.append(lemmatizer.lemmatize(word))
        lemma_sentence.append(" ")
    return "".join(lemma_sentence)
    # lemma_sentence = lemmatizeSentence("The circumstances under which I now meet you will acquit me from entering into that subject further than to refer to the great constitutional charter under which you are assembled, and which, in defining your powers, designates the objects to which your attention is to be given.")
    # print("The lemmatized sentence is: ", lemma_sentence)

#WILDCARD QUERY HANDLING
# Function that matches input strr with
# given wildcard pattern
def wildcard_handling(strr, pattern, n, m):
    # empty pattern can only match with
    # empty string
    if (m == 0):
        return (n == 0)
    # lookup table for storing results of
    # subproblems
    lookup = [[False for i in range(m + 1)] for j in range(n + 1)]
    # empty pattern can match with empty string
    lookup[0][0] = True
    # Only '*' can match with empty string
    for j in range(1, m + 1):
        if (pattern[j - 1] == '*'):
            lookup[0][j] = lookup[0][j - 1]
 
    # fill the table in bottom-up fashion
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Two cases if we see a '*'
            # a) We ignore ‘*’ character and move
            # to next character in the pattern,
            # i.e., ‘*’ indicates an empty sequence.
            # b) '*' character matches with ith
            # character in input
            if (pattern[j - 1] == '*'):
                lookup[i][j] = lookup[i][j - 1] or lookup[i - 1][j]
            # Current characters are considered as
            # matching in two cases
            # (a) current character of pattern is '?'
            # (b) characters actually match
            elif ((pattern[j-1]=='?')or(strr[i - 1] == pattern[j - 1])):
                lookup[i][j] = lookup[i - 1][j - 1]
 
            # If characters don't match
            else:
                lookup[i][j] = False
 
    #return lookup[n][m]
    return strr
 
pattern = "*****ba*****ab"
 
# if (wildcard_handling(strr, pattern, len(strr), len(pattern))):
#     print("Yes")
# else:
#     print("No")

#SPELLING CORRECTION: EDIT DISTANCE METHOD
def spelling_correction(strr):
    correct_words = words.words()
    # list of incorrect spellings
    # that need to be corrected
    incorrect_words=[]
    # loop for finding correct spellings
    # based on edit distance and
    # printing the correct words
    return strr

correct_words = words.words()
incorrect_words=[]
for word in incorrect_words:
	temp = [(edit_distance(word, w),w) for w in correct_words if w[0]==word[0]]
	print(sorted(temp, key = lambda val:val[0])[0][1])

final_words=[]
import os
#providing the path of the directory
#r = raw string literal
dirloc = r"C:\Users\shrey\Downloads\shakespeares-works_TXT_FolgerShakespeare" 
#calling scandir() function
for file in os.scandir(dirloc):
    if (file.path.endswith(".txt") and file.is_file()):
        #print(file.path)
        corpus = inaugural.raw(rf"{file.path}")
        corpus1 = stopword_removal(corpus)
        corpus2 = stemming(corpus1)
        corpus3 = lemmatization(corpus2)
        final_words.append(corpus3)

#print(final_words)

#################################################################################################################
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import sent_tokenize , word_tokenize
import glob
import re
import os
import numpy as np
import sys
Stopwords = set(stopwords.words('english'))

def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\s]'
    text = re.sub(regex,'',text)
    return text

all_words = []
dict_global = {}
file_folder = r'C:\Users\shrey\Downloads\shakespeares-works_TXT_FolgerShakespeare/*'
idx = 1
files_with_index = {}

def finding_all_unique_words_and_freq(words):
    words_unique = []
    word_freq = {}
    for word in words:
        if word not in words_unique:
            words_unique.append(word)
    for word in words_unique:
        word_freq[word] = words.count(word)
    return word_freq

for file in glob.glob(file_folder):
    #print(file)
    fname = file
    file = open(file , "r")
    text = file.read()
    text = remove_special_characters(text)
    text = re.sub(re.compile('\d'),'',text)
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words = [word for word in words if len(words)>1]
    words = [word.lower() for word in words]
    words = [word for word in words if word not in Stopwords]
    dict_global.update(finding_all_unique_words_and_freq(words))
    files_with_index[idx] = os.path.basename(fname)
    idx = idx + 1
    
unique_words_all = set(dict_global.keys())


def finding_freq_of_word_in_doc(word,words):
    freq = words.count(word)
        
def remove_special_characters(text):
    regex = re.compile('[^a-zA-Z0-9\s]')
    text_returned = re.sub(regex,'',text)
    return text_returned

class Node:
    def __init__(self ,docId, freq = None):
        self.freq = freq
        self.doc = docId
        self.nextval = None
    
class SlinkedList:
    def __init__(self ,head = None):
        self.head = head

inverted_index_data = {}
for word in unique_words_all:
    inverted_index_data[word] = SlinkedList()
    inverted_index_data[word].head = Node(1,Node)
word_freq_in_doc = {}
idx = 1
for file in glob.glob(file_folder):
    file = open(file, "r")
    text = file.read()
    text = remove_special_characters(text)
    text = re.sub(re.compile('\d'),'',text)
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words = [word for word in words if len(words)>1]
    words = [word.lower() for word in words]
    words = [word for word in words if word not in Stopwords]
    word_freq_in_doc = finding_all_unique_words_and_freq(words)
    for word in word_freq_in_doc.keys():
        inverted_index = inverted_index_data[word].head
        while inverted_index.nextval is not None:
            inverted_index = inverted_index.nextval
        inverted_index.nextval = Node(idx ,word_freq_in_doc[word])
    idx = idx + 1

query = input('Enter your query:')
query = word_tokenize(query)
#query = wildcard_handling(query, pattern, len(query), len(pattern))
#query = spelling_correction(query)
connecting_words = []
cnt = 1
different_words = []
for word in query:
    if word.lower() != "and" and word.lower() != "or" and word.lower() != "not":
        different_words.append(word.lower())
    else:
        connecting_words.append(word.lower())
print("Connecting words:", connecting_words)
total_files = len(files_with_index)
zeroes_and_ones = []
zeroes_and_ones_of_all_words = []
for word in (different_words):
    if word.lower() in unique_words_all:
        zeroes_and_ones = [0] * total_files
        linkedlist = inverted_index_data[word].head
        print(word)
        while linkedlist.nextval is not None:
            zeroes_and_ones[linkedlist.nextval.doc - 1] = 1
            linkedlist = linkedlist.nextval
        zeroes_and_ones_of_all_words.append(zeroes_and_ones)
    else:
        print(word,"shakespeares-works_TXT_FolgerShakespeare")
        sys.exit()
print(zeroes_and_ones_of_all_words)
for word in connecting_words:
    word_list1 = zeroes_and_ones_of_all_words[0]
    word_list2 = zeroes_and_ones_of_all_words[1]
    if word == "and":
        bitwise_op = [w1 & w2 for (w1,w2) in zip(word_list1,word_list2)]
        zeroes_and_ones_of_all_words.remove(word_list1)
        zeroes_and_ones_of_all_words.remove(word_list2)
        zeroes_and_ones_of_all_words.insert(0, bitwise_op)
    elif word == "or":
        bitwise_op = [w1 | w2 for (w1,w2) in zip(word_list1,word_list2)]
        zeroes_and_ones_of_all_words.remove(word_list1)
        zeroes_and_ones_of_all_words.remove(word_list2)
        zeroes_and_ones_of_all_words.insert(0, bitwise_op)
    elif word == "not":
        bitwise_op = [not w1 for w1 in word_list2]
        bitwise_op = [int(b == True) for b in bitwise_op]
        zeroes_and_ones_of_all_words.remove(word_list2)
        zeroes_and_ones_of_all_words.remove(word_list1)
        bitwise_op = [w1 & w2 for (w1,w2) in zip(word_list1,bitwise_op)]
#zeroes_and_ones_of_all_words.insert(0, bitwise_op)
        
files = []    
print(zeroes_and_ones_of_all_words)
lis = zeroes_and_ones_of_all_words[0]
cnt = 1
for index in lis:
    if index == 1:
        files.append(files_with_index[cnt])
    cnt = cnt+1    
print(files)

###############################################################################
# # program to compute the time
# import time
# # we initialize the variable start
# # to store the starting time of
# # execution of program
# start = time.time()
# # we can take any program but for
# # example we have taken the below
# # program
# a = 0
# for i in range(1000):
# 	a += (i**100)
# # now we have initialized the variable
# # end to store the ending time after
# # execution of program
# end = time.time()
# # difference of start and end variables
# # gives the time of execution of the
# # program in between
# print("The time of execution of above program is :", end-start)


