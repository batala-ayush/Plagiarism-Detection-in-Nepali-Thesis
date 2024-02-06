from .preprocessing import preprocess
import hashlib

def winnowing(text, k, w):
    """
    Winnowing algorithm for text fingerprinting using MD5 as a non-cryptographic hash.

    Parameters:
        text (str): The input text to be fingerprinted.
        k (int): Length of the shingles (substring length).
        w (int): Size of the winnowing window.

    Returns:
        set: Set of winnowed hash values forming the fingerprint.
    """
    def hash_shingle(shingle):
        md5_hash = hashlib.md5(shingle.encode('utf-8'))
        return md5_hash.hexdigest()

    def generate_shingles(text, k):
        return {text[i:i+k] for i in range(len(text) - k + 1)}

    def winnow(shingles, w):
        min_hash_positions = []
        shingles_list = list(shingles)
        for i in range(len(shingles) - w + 1):
            window = shingles_list[i:i+w]
            min_hash = min((hash_shingle(shingle) for shingle in window))
            min_hash_positions.append(min_hash)
        return min_hash_positions

    shingles = generate_shingles(text, k)
    winnowed_hashes = winnow(shingles, w)
    return winnowed_hashes

def generate_fingerprint_sets(text1):
    shingle_length = 5
    winnowing_window =5
    fingerprint_list= winnowing(" ".join(preprocess(text1)),shingle_length,winnowing_window)
    return fingerprint_list




def compute_fingerprint_similarity(list1,list2) :
    set1 = set(list1)
    set2 = set(list2)
    numerator = len(set1.intersection(set2))
    if len(set1)!=0:
        jac_similarity_mod = numerator/len(set1)
    else:
        jac_similarity_mod = 0
    result = round(jac_similarity_mod,3)
    return result