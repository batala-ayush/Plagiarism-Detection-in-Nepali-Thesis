from .preprocessing import preprocess

def generate_n_grams(text):
    tokens=list(preprocess(text))
    #ngrams_list = list(ngrams(tokens,2))
    ngrams_list = list()

    for i in range(0,len(tokens)-1):
        grams = (tokens[i],tokens[i+1])
        ngrams_list.append(grams)
        
    return ngrams_list 




def compute_ngram_similarity(ngrams1, ngrams2):
    # Convert the bigrams lists to sets
    set1 = set(ngrams1)
    set2_database = [tuple(bigrams) for bigrams in ngrams2]
    set2 = set(set2_database)
    #set2 = set(ngrams2)
    numerator = len(set1.intersection(set2))
    if len(set1)!=0:
        jac_similarity_mod = numerator/len(set1)
    else:
        jac_similarity_mod = 0
    result = round(jac_similarity_mod,3)
        
    return result




