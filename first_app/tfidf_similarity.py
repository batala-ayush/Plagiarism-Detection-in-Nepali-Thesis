#importing the library for TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
#import the libraries for SVD
from sklearn.decomposition import TruncatedSVD
#import library for cosine similarity
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math

from .preprocessing import preprocess




def sort_union(union):
    return sorted(union)

def all_union(sentences):
    union = list()
    union = list(set(sentences[0] + sentences[1]))
    return union

def all_tf(sentence,arr_list):
    tf = list()
    for word in arr_list:
        if word in sentence:
            value = sentence.count(word)/len(sentence)
            tf.append(round(value, 2))
        else:
            tf.append(0)
    return tf

def com_all_tf(sentences,arr_list):
    all_t = []
    for sentence in sentences:
        temp = all_tf(sentence,arr_list)
        all_t.append(temp)
    return all_t


def idf(t, documents):
    # Calculate the number of documents containing the term t
    n = sum(1 for document in documents if t in document)
    # Calculate the total number of documents in the corpus
    N = len(documents)
    # Calculate the IDF value for the term t
    return math.log((N + 1) / (n + 1)) + 1#math.log(N / n)

def ret_idf(arr_sen,sentences):
    idf_list = list()
    for word in arr_sen:
        val = idf(word, sentences)
        idf_list.append(round(val,2))
    return idf_list

def compute_tfidfsimilarity(p1,p2):
    preprocessed_sentences1 = preprocess(p1)
    preprocessed_sentences2 = preprocess(p2)
    sentences = [preprocessed_sentences1,preprocessed_sentences2]
    union_list = all_union(sentences)
    arr_list = sort_union(union_list)
    tf_mat = com_all_tf(sentences,arr_list)
    idf_mat = ret_idf(arr_list,sentences)
    
    tfidf_mat = []
    for row in tf_mat:
        temp = list()
        for i in range(len(row)):
            #print(i)
            val = (row[i])*idf_mat[i]
            temp.append(round(val,2))
        tfidf_mat.append(temp)
        
    # Compute cosine similarity
    array1 = tfidf_mat[0]
    array2 = tfidf_mat[1]

        ## calculation for the similarity
    numerator=np.dot(array1,array2)

    denom=np.sqrt(sum(np.square(array1)))*np.sqrt(sum(np.square(array2)))

        # Similarity:
    similarity = round(numerator/denom,2)
    return similarity




