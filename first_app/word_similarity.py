import pandas as pd
import numpy as np

data = pd.read_csv('first_app/book_posy.csv', encoding = 'utf-8')
data = data.fillna(method="ffill") # Deal with N/A
data_sample = data
import json
from pygtrie import Trie

# Load the JSON file
with open('first_app/output.json', encoding = 'utf-8') as file:
    data = json.load(file)

# Create a Trie data structure
trie = Trie()

 # Iterate over all data elements
for entry in data:
    word = entry['word']
    synonyms = entry['synonyms']
    # Insert the word and its qualifying senses into the Trie
    trie[word] = synonyms



tags_list = sorted(list(set(data_sample["tags"].values)))
words_dict = sorted(list(set(data_sample["words"].values)))
words_dict.append("OUT_OF_VOC")
words_dict.append("Padded_Value")
word2id = {w: i for i, w in enumerate(words_dict)}
tag2id = {t: i for i, t in enumerate(tags_list)}



from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

loaded_model = load_model("first_app/pos_model_new.h5")




suffixes = {'अगाडि','अगावै','अगि','अघि','अतिरिक्त','अनुकूल','अनुरुप','अनुरूप','अनुसार','अन्तरगत',
            'अन्तर्गत','अन्तर्गत्','अर्न्तगत','अलावा','उपर','उप्रान्त','कन','कहाँ','का','काँ','की',
            'कै','को','कों','खातिर','खेरि','गरिसकी','छेउ','जत्ति','जत्रा','जत्रै','जसमध्ये','जसै',
            'जसो','जस्ता','जस्ती','जस्तै','जस्तो','झै','झैँ','झैं','तक','तक्','तर्फ','तल','ताक',
            'ताका','तिर','तिरै','तिर्','तीर','थर','थरी','देखि','देखिन','देखिनै','देखिन्','देखी','द्वारा',
            'धरी','नगिचै','नजिक','नजिकै','निकट','निमित','निमित्त','निम्ति','निम्नबमोजिम','निम्नानुसार',
            'निर','निहित','नीर','नुसार','नेर','पछाडि','पछाडिपट्टि','पछि','पछी','पट्टि','पट्टी','पनि',
            'पर','परक','पर्यन्त','पश्चात','पश्चात्','पश्चिमपट्टि','पारि','पारिपट्टि','पारी','पिच्छे',
            'पूर्व','पूर्वक','प्रति','प्रभृति','बमोजिम','बाट','बाटै','बापत','बारे','बाहिर','बाहेक',
            'बिच','बित्तिकै','बिना','बीच','बेगर','भन्दा','भन्ने','भर','भरि','भरी','भित्र','भित्रै',
            'मध्ये','मनि','मन्','मा','माझ','माथि','मारे','मार्फत','मार्फत्','मुताविक','मुनि',
            'मूलक','मै','मैं','यता','रूपी','लगायत','लगि','लाइ','लाई','लाईं','लागि','लागी','ले',
            'ले.','वरिपरि','वस','वाट','वाट्','वापत','वारि','वारे','वाला','विच','वित्तिकै','विना',
            'विरुद्ध','विरूद्ध','सँग','सँगसँगै','सँगै','संग','संगै','सङ्ग','समक्ष','समेत','सम्बन्धि',
            'सम्बन्धी','सम्म','सम्मत','सम्मन','सम्मै','सम्वन्धी','सरह','सरि','सरी','सहित','साथ',
            'साथसाथै','साथै','सामु','सामुन्ने','सित','सिवाय','सीत','स्थित','स्वरूप','हर','हरु','हरू','हाले'}





# Create a dictionary to map specific NELRALEC tags to their generalized POS tags
pos_mapping = {
    # Noun
    'NN': 'Noun',
    'NP': 'Noun',
    
    # Pronoun
    'PMX': 'Pronoun',
    'PTN': 'Pronoun',
    'PTM': 'Pronoun',
    'PTH': 'Pronoun',
    'PXH': 'Pronoun',
    'PXR': 'Pronoun',
    'PMXKM': 'Pronoun',
    'PMXKF': 'Pronoun',
    'PMXKO': 'Pronoun',
    'PTNKM': 'Pronoun',
    'PTNKF': 'Pronoun',
    'PTNKO': 'Pronoun',
    'PTMKM': 'Pronoun',
    'PTMKF': 'Pronoun',
    'PTMKO': 'Pronoun',
    'PRFKM': 'Pronoun',
    'PRFKF': 'Pronoun',
    'PRFKO': 'Pronoun',
    'PMXKX': 'Pronoun',
    'PTNKX': 'Pronoun',
    'PTMKX': 'Pronoun',
    'PRFKX': 'Pronoun',
    'PRF': 'Pronoun',

    # Determiner
    'DDM': 'Determiner',
    'DDF': 'Determiner',
    'DKM': 'Determiner',
    'DKF': 'Determiner',
    'DJM': 'Determiner',
    'DJF': 'Determiner',
    'DGM': 'Determiner',
    'DGF': 'Determiner',
    'DDO': 'Determiner',
    'DKO': 'Determiner',
    'DJO': 'Determiner',
    'DGO': 'Determiner',
    'DDX': 'Determiner',
    'DKX': 'Determiner',
    'DJX': 'Determiner',
    'DGX': 'Determiner',
    'RD': 'Determiner',
    'RK': 'Determiner',
    'RJ': 'Determiner',
    
    # Verb
    'VVMX1': 'Verb',
    'VVMX2': 'Verb',
    'VVTN1': 'Verb',
    'VVTX2': 'Verb',
    'VVYN1': 'Verb',
    'VVYX2': 'Verb',
    'VVTN1F': 'Verb',
    'VVTM1F': 'Verb',
    'VVYN1F': 'Verb',
    'VVYM1F': 'Verb',
    'VOMX1': 'Verb',
    'VOMX2': 'Verb',
    'VOTN1': 'Verb',
    'VOTX2': 'Verb',
    'VOYN1': 'Verb',
    'aVOYX2': 'Verb',
    'VI': 'Verb',
    'VN': 'Verb',
    'VDM': 'Verb',
    'VDF': 'Verb',
    'VDO': 'Verb',
    'VDX': 'Verb',
    'VE': 'Verb',
    'VQ': 'Verb',
    'VCN': 'Verb',
    'VCM': 'Verb',
    'VCH': 'Verb',
    'VS': 'Verb',
    'VR': 'Verb',

    # Adjective
    'JM': 'Adjective',
    'JF': 'Adjective',
    'JO': 'Adjective',
    'JX': 'Adjective',
    'JT': 'Adjective',

    # Adverb
    'RR': 'Adverb',

    # Postposition
    'II': 'Postposition',
    'IH': 'Postposition',
    'IE': 'Postposition',
    'IA': 'Postposition',
    'IKM': 'Postposition',
    'IKO': 'Postposition',
    'IKF': 'Postposition',
    'IKX': 'Postposition',
    
    # Numerals
    'MM': 'Numerals',
    'MOM': 'Numerals',
    'MOF': 'Numerals',
    'MOO': 'Numerals',
    'MOX': 'Numerals',
    
    # Classifier
    'MLM': 'Classifier',
    'MLF': 'Classifier',
    'MLO': 'Classifier',
    'MLX': 'Classifier',
    
    # Conjunction
    'CC': 'Conjunction',
    'CSA': 'Conjunction',
    'CSB': 'Conjunction',
    
    # Interjection
    'UU': 'Interjection',
    
    # Question Marker
    'QQ': 'Question Marker',
    
    # Particle
    'TT': 'Particle',
    
    # Punctuation
    'YF': 'Punctuation',
    'YM': 'Punctuation',
    'YQ': 'Punctuation',
    'YB': 'Punctuation',
    
    # Foreign Word
    'FF': 'Foreign Word',
    'FS': 'Foreign Word',
    'FO': 'Foreign Word',
    'FZ': 'Foreign Word',
    
    # Unclassifiable
    'FU': 'Unclassifiable',
    
    # Abbreviation
    'FB': 'Abbreviation',
    
    # NULL Tag
    'NULL': 'NULL Tag',
}





stopwords_list = [
    'अक्सर', 'अगाडि', 'अझै', 'अनुसार', 'अन्तर्गत', 'अन्य', 'अन्यत्र', 'अन्यथा', 'अब', 'अरू',
    'अरूलाई', 'अर्को', 'अर्थात', 'अर्थात्', 'अलग', 'आए', 'आजको', 'आठ', 'आत्म', 'आदि',
    'आफू', 'आफूलाई', 'आफैलाई', 'आफ्नै', 'आफ्नो', 'आयो', 'उदाहरण', 'उन', 'उनको', 'उनले',
    'उप', 'उहाँलाई', 'एउटै', 'एक', 'एकदम', 'औं', 'कतै', 'कम', 'से', 'कम', 'कसरी', 'कसै',
    'कसैले', 'कहाँबाट', 'कहिलेकाहीं', 'कहिल्यै', 'कहीं', 'का', 'कि', 'किन', 'किनभने', 'कुनै',
    'कुरा', 'कृपया', 'के', 'केहि', 'केही', 'को', 'कोही', 'क्रमशः', 'गए', 'गरि', 'गरी', 'गरेका',
    'गरेको', 'गरेर', 'गरौं', 'गर्छ', 'गर्छु', 'गर्दै', 'गर्न', 'गर्नु', 'गर्नुपर्छ', 'गर्ने',
    'गर्यौं', 'गैर', 'चाँडै', 'चार', 'चाले', 'चाहनुहुन्छ', 'चाहन्छु', 'चाहिए', 'छ', 'छन्', 'छु',
    'छैन', 'छौँ', 'छौं', 'जताततै', 'जब', 'जबकि', 'जसको', 'जसबाट', 'जसमा', 'जसलाई', 'जसले', 'जस्तै',
    'जस्तो', 'जस्तोसुकै', 'जहाँ', 'जान', 'जाहिर', 'जुन', 'जे', 'जो', 'ठीक', 'त', 'तत्काल', 'तथा',
    'तदनुसार', 'तपाइँको', 'तपाईं', 'तर', 'तल', 'तापनि', 'तिनी', 'तिनीहरू', 'तिनीहरूको', 'तिनीहरूलाई',
    'तिनीहरूले', 'तिमी', 'तिर', 'ती', 'तीन', 'तुरुन्तै', 'तेस्रो', 'त्यसकारण', 'त्यसपछि', 'त्यसमा',
    'त्यसैले', 'त्यहाँ', 'त्यो', 'थिए', 'थिएन', 'थिएनन्', 'थियो', 'दिए', 'दिनुभएको', 'दिनुहुन्छ', 'दुई',
    'देख', 'देखि', 'देखिन्छ', 'देखियो', 'देखे', 'देखेको', 'देखेर', 'देख्न', 'दोश्रो', 'दोस्रो', 'धेरै',
    'न', 'नजिकै', 'नत्र', 'नयाँ', 'नि', 'निम्ति', 'निम्न', 'निम्नानुसार', 'निर्दिष्ट', 'नै', 'नौ', 'पक्का',
    'पक्कै', 'पछि', 'पछिल्लो', 'पटक', 'पनि', 'पर्छ', 'पर्थ्यो', 'पर्याप्त', 'पहिले', 'पहिलो', 'पहिल्यै',
    'पाँच', 'पाँचौं', 'पूर्व', 'प्रति', 'प्रत्येक', 'प्लस', 'फेरि', 'बने', 'बन्द', 'बन्न', 'बरु', 'बाटो',
    'बारे', 'बाहिर', 'बाहेक', 'बीच', 'बीचमा', 'भए', 'भएको', 'भन', 'भने', 'भने्', 'भन्छन्', 'भन्छु', 'भन्दा',
    'भन्नुभयो', 'भन्ने', 'भर', 'भित्र', 'भित्री', 'म', 'मलाई', 'मा', 'मात्र', 'माथि', 'मुख्य', 'मेरो', 'यति',
    'यथोचित', 'यदि', 'यद्यपि', 'यस', 'यसको', 'यसपछि', 'यसबाहेक', 'यसरी', 'यसो', 'यस्तो', 'यहाँ', 'यहाँसम्म',
    'या', 'यी', 'यो', 'र', 'रही', 'रहेका', 'रहेको', 'राखे', 'राख्छ', 'राम्रो', 'रूप', 'लगभग', 'लाई', 'लागि',
    'ले', 'वरिपरि', 'वास्तवमा', 'वाहेक', 'विरुद्ध', 'विशेष', 'शायद', 'सँग', 'सँगै', 'सक्छ', 'सट्टा', 'सधैं', 'सबै',
    'सबैलाई', 'समय', 'सम्भव', 'सम्म', 'सही', 'साँच्चै', 'सात', 'साथ', 'साथै', 'सायद', 'सारा', 'सो', 'सोध्न', 'सोही',
    'स्पष्ट', 'हरे', 'हरेक', 'हामी', 'हामीलाई', 'हाम्रो', 'हुँ', 'हुन', 'हुने', 'हुनेछ', 'हुन्', 'हुन्छ', 'हो', 'होइन',
    'होइनन्', 'होला', 'होस्',''
]

punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                    ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~','।','”','“','’','‘']



def sentence_tokenize(text):
    sentences = text.strip().split('।')
    sentences = [sentence.strip() + ' ।' for sentence in sentences if sentence.strip()]
    return sentences


def word_tokenize(sentences):
    tok_sentences = []
    punctuations = [',', ';', '?', '!', '—', '-', '.','’','‘']
    for sentence in sentences:
        for punctuation in punctuations:
            sentence = sentence.replace(punctuation, ' ')

        sentence = sentence.strip().split()
        tok_sentences.append(sentence)
    return tok_sentences




def stem_word(word):
    stem = word
    suffixes_found = []

    while True:
        found_suffix = False
        for suffix in suffixes:
            if stem.endswith(suffix):
                stem = stem[:len(stem) - len(suffix)]
                suffixes_found.append(suffix)
                found_suffix = True
                break

        if not found_suffix:
            break

    return stem, suffixes_found[::-1]



def stem_words(tok_sentences):
    stemmed_sentences = []
    for sentence in tok_sentences:
        stemmed_words = []

        for word in sentence:
            stemmed_word, suffixes_found = stem_word(word)
            if stemmed_word or suffixes_found:
                if stemmed_word:
                    stemmed_words.append(stemmed_word)
                if suffixes_found:
                    stemmed_words.extend(suffixes_found)
        
        stemmed_sentences.append(stemmed_words)
    return stemmed_sentences



def pos_tag(stemmed_sentences):
    X_samp = [[word2id.get(w, word2id['OUT_OF_VOC']) for w in s] for s in stemmed_sentences]
    X_Samp = pad_sequences(maxlen=50, sequences=X_samp, padding="post",value =word2id["Padded_Value"])
    p1 = loaded_model.predict(np.array(X_Samp)) # Predict on it
    pr = np.argmax(p1, axis=-1) # Map softmax back to a POS index
    tagged_sentences = []

    for sentence, pred in zip(stemmed_sentences, pr):
        tagged_sentence = []
        for w, t in zip(sentence, pred):
            tagged_sentence.append((w, pos_mapping.get(tags_list[t], 'Unclassifiable')))
        tagged_sentences.append(tagged_sentence)
    return tagged_sentences




def merge_sentences(tagged_sentences):
    paragraph = []
    for sentence in tagged_sentences:
        paragraph.extend(sentence)
    return paragraph




def search_synonyms(word):
    if word in trie:
        return trie.get(word, None)
def remove_spaces_from_list_elements(input_list):
    return [word.replace(' ', '') for word in input_list if word]

def get_all_synonyms(search_word):

    synonyms = search_synonyms(search_word)
    all_synonyms = []
    

    if synonyms is None:
        return [search_word]
    else:
        all_synonyms.extend(synonyms)
        synonyms = remove_spaces_from_list_elements(synonyms)
        for word in synonyms:
            if word:
                synonyms_list = trie.get(word, None)
                # Print the word and its synonyms
                if synonyms_list is not None:
                    all_synonyms.extend(synonyms_list)

        all_synonyms = remove_spaces_from_list_elements(all_synonyms)
        all_synonyms.append(search_word)
        all_synonyms = list(set(all_synonyms))

    return all_synonyms



def create_original_wordlist(filtered_ori_paragraph):
    original_words_list = list()
    for word,pos in filtered_ori_paragraph:
        if pos in ['Noun','Adjective','Adverb']:
            original_words_list.append(word)
    orginal_words_list = list(set(original_words_list))
    return orginal_words_list



def modify_suspicious_paragraph(filtered_sus_paragraph,orginal_words_list):
    modified_sus_paragraph = list()
    for word,pos in filtered_sus_paragraph:
        is_modified = False
        if pos in ['Noun','Adjective','Adverb']:
            """for orig_word in orginal_words_list:
                if orig_word in get_all_synonyms(word):
                    #print(word,":",get_all_synonyms(word))
                    modified_sus_paragraph.append((orig_word,pos))
                    is_modified = True
                    break
            if is_modified == False:
                modified_sus_paragraph.append((word,pos))"""
            synonyms_words = get_all_synonyms(word)
            for syn in synonyms_words:
                if syn in orginal_words_list:
                    modified_sus_paragraph.append((syn,pos))
                    is_modified = True
                    break
            if is_modified == False:
                modified_sus_paragraph.append((word,pos))
        else:
            modified_sus_paragraph.append((word,pos))
    return modified_sus_paragraph




def jaccard_similarity_modified(paragraph1, paragraph2):
    s1 = set(paragraph1)
    s2 = set(paragraph2)
    if len(s1)!=0:
        jac_similarity_mod = len(s1.intersection(s2))/len(s1)
    else:
        jac_similarity_mod = 0
    result = round(jac_similarity_mod,3)
    return result


def generate_word_sim_sets(paragraph):
    sentences = sentence_tokenize(paragraph)
    
    tok_sentences = word_tokenize(sentences)
    
    stemmed_tok_sentences = stem_words(tok_sentences)
    
    tagged_sentences = pos_tag(stemmed_tok_sentences)
    
    sus_paragraph = merge_sentences(tagged_sentences)
    
    filtered_paragraph = [(word, pos) for word, pos in sus_paragraph if word not in stopwords_list and word not in punctuations]

    return filtered_paragraph





def compute_word_similarity(filtered_sus_paragraph,filtered_ori_paragraph_database):
    filtered_ori_paragraph = [tuple(element) for element in filtered_ori_paragraph_database ]
    orginal_words_list = create_original_wordlist(filtered_ori_paragraph)
    modified_sus_paragraph = modify_suspicious_paragraph(filtered_sus_paragraph,orginal_words_list)
    print(modified_sus_paragraph)
    jac_similarity_mod = jaccard_similarity_modified(modified_sus_paragraph,filtered_ori_paragraph)
    result = round(jac_similarity_mod,3)
    return result