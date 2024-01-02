#importing the library for TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
#import the libraries for SVD
from sklearn.decomposition import TruncatedSVD
#import library for cosine similarity
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math



stopwords_list = ['','सं', 'यहाँसम्', 'यहाँसम्म', 'आफैँ', 'उनले', 'त्यसम', 'जसल', 'गयौ', 'धन्न', 'निकै', 'पर्थ्य', 'राख्छ', 'आफू', 'तिमीसँ', 'ए', 'नत्', 'अन्तर्ग', 'दर्ता', 'त', 'लागि', 'आफ्नै', 'गर्', 'क्रमश', 'भयेन', 'तापन', 'यत', 'तिनीहरूल', 'भन्', 'यथोचि', 'आज', 'फरक', 'ल', 'उ', 'आएका', 'द्वारा', 'यस्तो', 'रही', 'सोचेको', 'रुपमा', 'उनि', 'कारण', 'चाहनुहुन्छ', 'दोस्रो', 'तत्का', 'हुनेछ', 'दिएको', 'तेह्र', 'अन्यत्', 'या', 'मला', 'सबै', 'तथ', 'राम्रो', 'सबैको', 'भा', 'उक्त', 'आदि', 'पहिल', 'पछिल्ल', 'हुन्थ', 'र', 'दे', 'गर्नुपर्', 'दिनुहुन्', 'त्य', 'दोस्र', 'दि', 'चोटी', 'दिए', 'अन्तर्गत', 'कसैको', 'तापनि', 'हाम्रा', 'गर्यौ', 'पर्ने', 'आफ', 'तेस्र', 'सँग', 'हुन', 'जस्तै', 'बन', 'तपाईको', 'ती', 'हुँदै', 'थिएन', 'भनेको', 'उनल', 'जसलाई', 'तिनीहरूको', 'हुन्', 'प्रत', 'धौ', 'कोहिपनि', 'औं', 'कोहीपनि', 'घर', 'पहिल्यै', 'संग', 'यद्यप', 'मध्ये', 'यसक', 'गरेको', 'सम', 'जहा', 'पहिल्य', 'भयो', 'जसबाट', 'थिए', 'जो', 'अरू', 'रे', 'कहिलेकाहीं', 'सा', 'सारा', 'ले', 'तिनिहरुलाई', 'पांच', 'त्यस्तो', 'जसम', 'प्लस', 'भने', 'भन्न', 'तपा', 'जनाको', 'अरुलाई', 'भएक', 'सहितै', 'गर्द', 'भन्द', 'किन', 'छन्', 'तुरुन्तै', 'साथै', 'बाट', 'त्यसले', 'सङ्ग', 'आफूला', 'तथापि', 'त्यसैले', 'विशे', 'तीन', 'उसलाई', 'जुन', 'निर्दिष्ट', 'तिर', 'लाख', 'कोहि', 'यस्त', 'यसपछ', 'माथी', 'आजको', 'यस', 'उन्नाइस', 'चाहेको', 'पछाडी', 'बिशे', 'भएर', 'राम्र', 'के', 'पिच्छे', 'कुनै', 'अठार', 'बार', 'अरु', 'उन', 'जता', 'थि', 'राखे', 'दुब', 'धेर', 'त्यसपछ', 'आदिलाई', 'भए', 'तेस्रो', 'आए', 'गर्छ', 'भएन', 'जसक', 'प्रते', 'सबैलाई', 'यस्तै', 'अझै', 'थरि', 'उनको', 'हर', 'सोध्', 'ग', 'यसले', 'अलग', 'सत्र', 'भित्र', 'फेरि', 'गै', 'पर्छ', 'निम्ति', 'आफनो', 'देखेको', 'सहित', 'मेरो', 'सरह', 'भरि', 'माथ', 'राख', 'अनुसा', 'आफ्नो', 'एकदम', 'ठाउँमा', 'न', 'तिनै', 'सँगसँगै', 'कहिल्य', 'व', 'सक्', 'तिनी', 'अर्था', 'नि', 'हामीसँ', 'तैपनी', 'नगर्नु', 'नभनेर', 'छैनन्', 'सम्भ', 'नयाँ', 'आएको', 'यहाँ', 'तिनीहरूला', 'रहेछ', 'सँगको', 'बताए', 'थियो', 'लाग', 'तता', 'त्सपछ', 'पन्ध्र', 'रह', 'जोपनि', 'सकदि', 'छु', 'चाहि', 'जस्तोसुक', 'जबकी', 'सङ्गको', 'भनेर', 'देखि', 'ऊ', 'सही', 'चाहिए', 'बिच', 'उनलाई', 'त्यहाँ', 'कता', 'सँ', 'ह', 'एउटा', 'आफैल', 'गरी', 'साथ', 'देखेक', 'गरेक', 'बीचम', 'मुन्तिर', 'कस्त', 'भन्छन', 'लाई', 'गएको', 'हुनु', 'एउट', 'यथोचित', 'मध्य', 'उस्त', 'पक्कै', 'भन्ने', 'यो', 'समयम', 'आफूलाई', 'उदाहरण', 'नजिक', 'का', 'यद', 'तिरको', 'यसबाहेक', 'उहा', 'कसैसँग', 'कि', 'गरौं', 'बाह्र', 'प्रतेक', 'विरुद्', 'वटा', 'हुन्थ्यो', 'हाम्र', 'थिएनन', 'कतै', 'कुन', 'पक्क', 'नजिकै', 'त्यहीँ', 'सब', 'लागेको', 'कस', 'भन्नुभय', 'यद्ध्यपि', 'बने', 'कृपया', 'शायद', 'धेरै', 'उनी', 'कुन्नी', 'नत्रभने', 'हुने', 'थरी', 'मात्रै', 'चाहीं', 'साय', 'अघि', 'ला', 'केह', 'अ', 'थिय', 'यसको', 'सोह्र', 'तेश्रो', 'पो', 'सक्दै', 'चाहन्छु', 'सक्ने', 'जे', 'निम्', 'अवस्था', 'कसरी', 'तेस्कारण', 'दिनुभएको', 'भएँ', 'तिनीहरुक', 'कहा', 'आजक', 'उप', 'बीच', 'थ', 'देखिन्छ', 'जसरी', 'अन्यत्र', 'बन्', 'बी', 'भएकालाई', 'गरेर', 'जुनै', 'चाल', 'पाचौँ', 'तर्फ', 'अरुला', 'अझ', 'म', 'कसै', 'जनाले', 'गैर', 'सम्भव', 'हरे', 'सुरु', 'त्यत्तिकैमा', 'बिचमा', 'देखिय', 'तैपनि', 'सोचेर', 'सुनेर', 'तेस्कार', 'तिनीहरू', 'नै', 'यसो', 'बाहिर', 'पाँचौ', 'पर्याप्त', 'निम्न', 'जना', 'हाम्रो', 'त्यो', 'कहिल', 'मलाई', 'उनीहरुको', 'मुख्य', 'हामी', 'देखेर', 'उनीहर', 'यहा', 'वरीपरी', 'त्सपछि', 'कसैल', 'जस्तोसुकै', 'मा', 'जाहिर', 'मुनि', 'पाँचौं', 'अर्क', 'तिनि', 'चाँड', 'आफैला', 'तर', 'पर्नेमा', 'संगै', 'पनि', 'कसैले', 'देखिन्', 'देखे', 'देख्', 'त्यसको', 'भन्नुभयो', 'भएका', 'काम', 'कसैलाई', 'स्पष्', 'भन्दा', 'प्ल', 'अक्स', 'नत्र', 'त्सैल', 'फेरी', 'झैं', 'तब', 'पछी', 'रू', 'स', 'कहिलेकाही', 'अगाडि', 'गर्ने', 'सम्म', 'जबक', 'जसबा', 'पहिले', 'भन', 'तदनुसार', 'नभएको', 'तपाई', 'जनालाई', 'समय', 'थप', 'केही', 'भन्छन्', 'तपाईक', 'अलि', 'यसका', 'छू', 'क', 'अनुसार', 'आद', 'अहिले', 'बरु', 'तसरी', 'को', 'कुरा', 'रहेका', 'त्यस्तै', 'ज', 'तिम्र', 'गरे', 'छैन', 'फेर', 'यसमा', 'वा', 'गरेका', 'बिरुद्', 'आय', 'गर्छु', 'त्यहिँ', 'अर्थात', 'हैन', 'चाहेर', 'हामीले', 'हुँदैन', 'दु', 'अनि', 'अन्यथ', 'पर्दैन', 'बिशेष', 'पर्थ्यो', 'हुनुहुन्', 'जा', 'छन', 'उहाला', 'जाने', 'देख', 'त्यस', 'हुनत', 'अब', 'तथा', 'सधै', 'हुन्छ', 'वरीपर', 'एकद', 'सोह', 'ओठ', 'कम से क', 'किनभने', 'चौथो', 'रहेको', 'हजार', 'होल', 'नगर्नुहोस', 'बिरुद्ध', 'दोश्री', 'स्थित', 'आफै', 'रूप', 'अन्य', 'जसले', 'लगायत', 'जोपनी', 'तथापी', 'गर्दै', 'साँच्च', 'जस्ता', 'वास्तवम', 'त्यसकार', 'यसबाहे', 'वाहे', 'पछिल्लो', 'अन्यथा', 'देखियो', 'जु', 'निर्दिष्', 'अर्थात्', 'कति', 'भन्छु', 'सुरुमै', 'अक्सर', 'तिनीहरूक', 'तल', 'लगभग', 'होइ', 'सो', 'सकिए', 'पछि', 'यति', 'मैल', 'तुरन्त', 'जसको', 'जस्त', 'त्यहा', 'तिमी', 'गरौ', 'भित्', 'प्रत्ये', 'लगभ', 'दोश्र', 'गर्न', 'आफ्न', 'औ', 'कमसेकम', 'वास्तवमा', 'मुख्', 'पछाडि', 'अरूला', 'गए', 'राख्', 'ठूलो', 'त्यसो', 'गर्नेछ', 'गर्नेछन', 'यसपछि', 'स्पष्ट', 'जस्तो', 'होइनन', 'त्यही', 'दिनुभएक', 'थिएँ', 'थाहा', 'पक्का', 'भ', 'य', 'कही', 'हु', 'अर', 'चौध', 'त्यहीं', 'आत्', 'होला', 'हाम', 'पर्याप्', 'जसला', 'भीत्र', 'उसला', 'सध', 'बाहे', 'बढी', 'भन्या', 'बा', 'यही', 'चाहिंले', 'वापत', 'होस', 'भरी', 'गर्नु', 'केव', 'सट्टा', 'जतातत', 'निम्त', 'होइन', 'सार', 'गर्नुपर्छ', 'सुरुको', 'द्वार', 'वाट', 'दुइ', 'गर', 'अर्को', 'त्यसकारण', 'कस्तो', 'त्यत्तिकै', 'बिस', 'पट', 'गएर', 'छै', 'हरेक', 'एउटै', 'भन्छ', 'पर्', 'अगाडी', 'अगाड', 'दुइवटा', 'कहाँबाट', 'कसर', 'चाहनुहुन्', 'निम्नानुसा', 'सुनेको', 'यसैल', 'पन', 'यदि', 'ठी', 'भर', 'अल', 'पछ', 'बारेम', 'आयो', 'गर्दा', 'जबकि', 'वरिपर', 'चाले', 'ठीक', 'तुरुन्त', 'मेर', 'कोही', 'चाहन्छौ', 'कहाँबा', 'जाहि', 'तिनीहरु', 'यसरी', 'निम्नानुसार', 'चाहन्छ', 'त्सैले', 'उसले', 'कम', 'जसमा', 'जहाँ', 'नगर्नू', 'जान', 'तपाइँक', 'एघार', 'माथि', 'गरि', 'मात्र', 'कोह', 'उसको', 'प्रति', 'बाहेक', 'सँगै', 'हो', 'दोश्रो', 'चाहिं', 'बारे', 'त्यसैल', 'यसर', 'उनक', 'दिन', 'किनभन', 'चा', 'बर', 'तापनी', 'जताततै', 'उहालाई', 'सक्छ', 'नभई', 'भित्री', 'अधि', 'अन्', 'तिनिहरुला', 'कसरि', 'कत', 'जब', 'शाय', 'तदनुसा', 'पाँ', 'यद्यपि', 'सट्ट', 'बीस', 'यी', 'सोही', 'मात्', 'चाहन्थ', 'बीचमा', 'मार्फ', 'सह', 'छौ', 'साँच्चै', 'तत्काल', 'नया', 'सबैला', 'ति', 'आ', 'तपाइ', 'सायद', 'पहिलो', 'प्राय', 'ओ', 'सम्', 'दिनुहुन्छ', 'पटक', 'हामीला', 'छौं', 'बाहि', 'गय', 'समेत', 'होकि', 'उहाँला', 'प्रत्यक', 'अथवा', 'भन्दै', 'मैले', 'कुर', 'क्रमशः', 'तिम', 'कृपय', 'तिनीहर', 'पूर्व', 'उदाहर', 'भएको', 'रहेक', 'सय', 'केहि', 'गर्छन्', 'आत्म', 'पूर्', 'तिनीहरुको']

import string
def word_tokenize(text):
    punctuations = list(string.punctuation)
    # Add the Nepali purnabiram to the list of punctuations
    punctuations.append("।")
    punctuations.append("”")
    punctuations.append("“")
    punctuations.append("’")
    punctuations.append("‘")
    for punctuation in punctuations:
        text = text.replace(punctuation, ' ')
    text = text.strip().split()
    return text

import snowballstemmer
def word_stemmer(tokens):
    stemmer = snowballstemmer.NepaliStemmer()
    if isinstance(tokens, str):
        tokens = tokens.split()
    stemmed_tokens = [stemmer.stemWord(token) for token in tokens]
    return stemmed_tokens

def stopword_removal(tokens):
    tokens = tokens
    #f = open('stopwords_ne.txt','r',encoding='utf-8')
    #stopwords_list = f.read()
    new_tokens = list() #new_tokens hold the list of words after removing stopwords
    for token in tokens:
        if token not in stopwords_list:
            new_tokens.append(token)

    return new_tokens

import re

def remove_english_from_nepali(nepali_text):
    # Define regular expression to match English words and characters
    english_pattern = re.compile(r'[a-zA-Z]+')
    
    # Remove English words and characters from the Nepali text
    nepali_text = re.sub(english_pattern, '', nepali_text)
    
    # Return the cleaned Nepali text
    return nepali_text

def preprocess(text):
    tokens = word_tokenize(text)
    stemmed_tokens = word_stemmer(tokens)
    removed_tokens=stopword_removal(stemmed_tokens)
    return removed_tokens


def preprocessing(sentences):
    p1= sentences[0]
    p2 = sentences[1]
    tokens1 = word_tokenize(p1)
    stemmed_tokens1 = word_stemmer(tokens1)
    removed_tokens1=stopword_removal(stemmed_tokens1)
    tokens2 = word_tokenize(p2)
    stemmed_tokens2 = word_stemmer(tokens2)
    removed_tokens2 =stopword_removal(stemmed_tokens2)
    str1=""
    str2=""
    for ele in removed_tokens1:
        str1 = str1 + ele +" "

    for ele in removed_tokens2:
        str2 = str2 + ele +" "
    docs = [str1,str2]
    return docs





def sort_union(union):
    return sorted(union)

def all_union(sentences):
    union = list()
    for sen in sentences:
        sen = sen.strip().split()
        for word in sen:
            if word not in union:
                union.append(word)
    return union

def all_tf(sentence,arr_list):
    tf = list()
    sentence = sentence.split()
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

def calculate_tfidfsimilarity(p1,p2):
    sentences=[p1,p2]
    sentences=preprocessing(sentences)
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




