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





stopwords_list = ['','सं', 'यहाँसम्', 'यहाँसम्म', 'आफैँ', 'उनले', 'त्यसम', 'जसल', 'गयौ', 'धन्न', 'निकै', 'पर्थ्य', 'राख्छ', 'आफू', 'तिमीसँ', 'ए', 'नत्', 'अन्तर्ग', 'दर्ता', 'त', 'लागि', 'आफ्नै', 'गर्', 'क्रमश', 'भयेन', 'तापन', 'यत', 'तिनीहरूल', 'भन्', 'यथोचि', 'आज', 'फरक', 'ल', 'उ', 'आएका', 'द्वारा', 'यस्तो', 'रही', 'सोचेको', 'रुपमा', 'उनि', 'कारण', 'चाहनुहुन्छ', 'दोस्रो', 'तत्का', 'हुनेछ', 'दिएको', 'तेह्र', 'अन्यत्', 'या', 'मला', 'सबै', 'तथ', 'राम्रो', 'सबैको', 'भा', 'उक्त', 'आदि', 'पहिल', 'पछिल्ल', 'हुन्थ', 'र', 'दे', 'गर्नुपर्', 'दिनुहुन्', 'त्य', 'दोस्र', 'दि', 'चोटी', 'दिए', 'अन्तर्गत', 'कसैको', 'तापनि', 'हाम्रा', 'गर्यौ', 'पर्ने', 'आफ', 'तेस्र', 'सँग', 'हुन', 'जस्तै', 'बन', 'तपाईको', 'ती', 'हुँदै', 'थिएन', 'भनेको', 'उनल', 'जसलाई', 'तिनीहरूको', 'हुन्', 'प्रत', 'धौ', 'कोहिपनि', 'औं', 'कोहीपनि', 'घर', 'पहिल्यै', 'संग', 'यद्यप', 'मध्ये', 'यसक', 'गरेको', 'सम', 'जहा', 'पहिल्य', 'भयो', 'जसबाट', 'थिए', 'जो', 'अरू', 'रे', 'कहिलेकाहीं', 'सा', 'सारा', 'ले', 'तिनिहरुलाई', 'पांच', 'त्यस्तो', 'जसम', 'प्लस', 'भने', 'भन्न', 'तपा', 'जनाको', 'अरुलाई', 'भएक', 'सहितै', 'गर्द', 'भन्द', 'किन', 'छन्', 'तुरुन्तै', 'साथै', 'बाट', 'त्यसले', 'सङ्ग', 'आफूला', 'तथापि', 'त्यसैले', 'विशे', 'तीन', 'उसलाई', 'जुन', 'निर्दिष्ट', 'तिर', 'लाख', 'कोहि', 'यस्त', 'यसपछ', 'माथी', 'आजको', 'यस', 'उन्नाइस', 'चाहेको', 'पछाडी', 'बिशे', 'भएर', 'राम्र', 'के', 'पिच्छे', 'कुनै', 'अठार', 'बार', 'अरु', 'उन', 'जता', 'थि', 'राखे', 'दुब', 'धेर', 'त्यसपछ', 'आदिलाई', 'भए', 'तेस्रो', 'आए', 'गर्छ', 'भएन', 'जसक', 'प्रते', 'सबैलाई', 'यस्तै', 'अझै', 'थरि', 'उनको', 'हर', 'सोध्', 'ग', 'यसले', 'अलग', 'सत्र', 'भित्र', 'फेरि', 'गै', 'पर्छ', 'निम्ति', 'आफनो', 'देखेको', 'सहित', 'मेरो', 'सरह', 'भरि', 'माथ', 'राख', 'अनुसा', 'आफ्नो', 'एकदम', 'ठाउँमा', 'न', 'तिनै', 'सँगसँगै', 'कहिल्य', 'व', 'सक्', 'तिनी', 'अर्था', 'नि', 'हामीसँ', 'तैपनी', 'नगर्नु', 'नभनेर', 'छैनन्', 'सम्भ', 'नयाँ', 'आएको', 'यहाँ', 'तिनीहरूला', 'रहेछ', 'सँगको', 'बताए', 'थियो', 'लाग', 'तता', 'त्सपछ', 'पन्ध्र', 'रह', 'जोपनि', 'सकदि', 'छु', 'चाहि', 'जस्तोसुक', 'जबकी', 'सङ्गको', 'भनेर', 'देखि', 'ऊ', 'सही', 'चाहिए', 'बिच', 'उनलाई', 'त्यहाँ', 'कता', 'सँ', 'ह', 'एउटा', 'आफैल', 'गरी', 'साथ', 'देखेक', 'गरेक', 'बीचम', 'मुन्तिर', 'कस्त', 'भन्छन', 'लाई', 'गएको', 'हुनु', 'एउट', 'यथोचित', 'मध्य', 'उस्त', 'पक्कै', 'भन्ने', 'यो', 'समयम', 'आफूलाई', 'उदाहरण', 'नजिक', 'का', 'यद', 'तिरको', 'यसबाहेक', 'उहा', 'कसैसँग', 'कि', 'गरौं', 'बाह्र', 'प्रतेक', 'विरुद्', 'वटा', 'हुन्थ्यो', 'हाम्र', 'थिएनन', 'कतै', 'कुन', 'पक्क', 'नजिकै', 'त्यहीँ', 'सब', 'लागेको', 'कस', 'भन्नुभय', 'यद्ध्यपि', 'बने', 'कृपया', 'शायद', 'धेरै', 'उनी', 'कुन्नी', 'नत्रभने', 'हुने', 'थरी', 'मात्रै', 'चाहीं', 'साय', 'अघि', 'ला', 'केह', 'अ', 'थिय', 'यसको', 'सोह्र', 'तेश्रो', 'पो', 'सक्दै', 'चाहन्छु', 'सक्ने', 'जे', 'निम्', 'अवस्था', 'कसरी', 'तेस्कारण', 'दिनुभएको', 'भएँ', 'तिनीहरुक', 'कहा', 'आजक', 'उप', 'बीच', 'थ', 'देखिन्छ', 'जसरी', 'अन्यत्र', 'बन्', 'बी', 'भएकालाई', 'गरेर', 'जुनै', 'चाल', 'पाचौँ', 'तर्फ', 'अरुला', 'अझ', 'म', 'कसै', 'जनाले', 'गैर', 'सम्भव', 'हरे', 'सुरु', 'त्यत्तिकैमा', 'बिचमा', 'देखिय', 'तैपनि', 'सोचेर', 'सुनेर', 'तेस्कार', 'तिनीहरू', 'नै', 'यसो', 'बाहिर', 'पाँचौ', 'पर्याप्त', 'निम्न', 'जना', 'हाम्रो', 'त्यो', 'कहिल', 'मलाई', 'उनीहरुको', 'मुख्य', 'हामी', 'देखेर', 'उनीहर', 'यहा', 'वरीपरी', 'त्सपछि', 'कसैल', 'जस्तोसुकै', 'मा', 'जाहिर', 'मुनि', 'पाँचौं', 'अर्क', 'तिनि', 'चाँड', 'आफैला', 'तर', 'पर्नेमा', 'संगै', 'पनि', 'कसैले', 'देखिन्', 'देखे', 'देख्', 'त्यसको', 'भन्नुभयो', 'भएका', 'काम', 'कसैलाई', 'स्पष्', 'भन्दा', 'प्ल', 'अक्स', 'नत्र', 'त्सैल', 'फेरी', 'झैं', 'तब', 'पछी', 'रू', 'स', 'कहिलेकाही', 'अगाडि', 'गर्ने', 'सम्म', 'जबक', 'जसबा', 'पहिले', 'भन', 'तदनुसार', 'नभएको', 'तपाई', 'जनालाई', 'समय', 'थप', 'केही', 'भन्छन्', 'तपाईक', 'अलि', 'यसका', 'छू', 'क', 'अनुसार', 'आद', 'अहिले', 'बरु', 'तसरी', 'को', 'कुरा', 'रहेका', 'त्यस्तै', 'ज', 'तिम्र', 'गरे', 'छैन', 'फेर', 'यसमा', 'वा', 'गरेका', 'बिरुद्', 'आय', 'गर्छु', 'त्यहिँ', 'अर्थात', 'हैन', 'चाहेर', 'हामीले', 'हुँदैन', 'दु', 'अनि', 'अन्यथ', 'पर्दैन', 'बिशेष', 'पर्थ्यो', 'हुनुहुन्', 'जा', 'छन', 'उहाला', 'जाने', 'देख', 'त्यस', 'हुनत', 'अब', 'तथा', 'सधै', 'हुन्छ', 'वरीपर', 'एकद', 'सोह', 'ओठ', 'कम से क', 'किनभने', 'चौथो', 'रहेको', 'हजार', 'होल', 'नगर्नुहोस', 'बिरुद्ध', 'दोश्री', 'स्थित', 'आफै', 'रूप', 'अन्य', 'जसले', 'लगायत', 'जोपनी', 'तथापी', 'गर्दै', 'साँच्च', 'जस्ता', 'वास्तवम', 'त्यसकार', 'यसबाहे', 'वाहे', 'पछिल्लो', 'अन्यथा', 'देखियो', 'जु', 'निर्दिष्', 'अर्थात्', 'कति', 'भन्छु', 'सुरुमै', 'अक्सर', 'तिनीहरूक', 'तल', 'लगभग', 'होइ', 'सो', 'सकिए', 'पछि', 'यति', 'मैल', 'तुरन्त', 'जसको', 'जस्त', 'त्यहा', 'तिमी', 'गरौ', 'भित्', 'प्रत्ये', 'लगभ', 'दोश्र', 'गर्न', 'आफ्न', 'औ', 'कमसेकम', 'वास्तवमा', 'मुख्', 'पछाडि', 'अरूला', 'गए', 'राख्', 'ठूलो', 'त्यसो', 'गर्नेछ', 'गर्नेछन', 'यसपछि', 'स्पष्ट', 'जस्तो', 'होइनन', 'त्यही', 'दिनुभएक', 'थिएँ', 'थाहा', 'पक्का', 'भ', 'य', 'कही', 'हु', 'अर', 'चौध', 'त्यहीं', 'आत्', 'होला', 'हाम', 'पर्याप्', 'जसला', 'भीत्र', 'उसला', 'सध', 'बाहे', 'बढी', 'भन्या', 'बा', 'यही', 'चाहिंले', 'वापत', 'होस', 'भरी', 'गर्नु', 'केव', 'सट्टा', 'जतातत', 'निम्त', 'होइन', 'सार', 'गर्नुपर्छ', 'सुरुको', 'द्वार', 'वाट', 'दुइ', 'गर', 'अर्को', 'त्यसकारण', 'कस्तो', 'त्यत्तिकै', 'बिस', 'पट', 'गएर', 'छै', 'हरेक', 'एउटै', 'भन्छ', 'पर्', 'अगाडी', 'अगाड', 'दुइवटा', 'कहाँबाट', 'कसर', 'चाहनुहुन्', 'निम्नानुसा', 'सुनेको', 'यसैल', 'पन', 'यदि', 'ठी', 'भर', 'अल', 'पछ', 'बारेम', 'आयो', 'गर्दा', 'जबकि', 'वरिपर', 'चाले', 'ठीक', 'तुरुन्त', 'मेर', 'कोही', 'चाहन्छौ', 'कहाँबा', 'जाहि', 'तिनीहरु', 'यसरी', 'निम्नानुसार', 'चाहन्छ', 'त्सैले', 'उसले', 'कम', 'जसमा', 'जहाँ', 'नगर्नू', 'जान', 'तपाइँक', 'एघार', 'माथि', 'गरि', 'मात्र', 'कोह', 'उसको', 'प्रति', 'बाहेक', 'सँगै', 'हो', 'दोश्रो', 'चाहिं', 'बारे', 'त्यसैल', 'यसर', 'उनक', 'दिन', 'किनभन', 'चा', 'बर', 'तापनी', 'जताततै', 'उहालाई', 'सक्छ', 'नभई', 'भित्री', 'अधि', 'अन्', 'तिनिहरुला', 'कसरि', 'कत', 'जब', 'शाय', 'तदनुसा', 'पाँ', 'यद्यपि', 'सट्ट', 'बीस', 'यी', 'सोही', 'मात्', 'चाहन्थ', 'बीचमा', 'मार्फ', 'सह', 'छौ', 'साँच्चै', 'तत्काल', 'नया', 'सबैला', 'ति', 'आ', 'तपाइ', 'सायद', 'पहिलो', 'प्राय', 'ओ', 'सम्', 'दिनुहुन्छ', 'पटक', 'हामीला', 'छौं', 'बाहि', 'गय', 'समेत', 'होकि', 'उहाँला', 'प्रत्यक', 'अथवा', 'भन्दै', 'मैले', 'कुर', 'क्रमशः', 'तिम', 'कृपय', 'तिनीहर', 'पूर्व', 'उदाहर', 'भएको', 'रहेक', 'सय', 'केहि', 'गर्छन्', 'आत्म', 'पूर्', 'तिनीहरुको']
#need to add new stopwords list . This list isnt accurate


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
            for orig_word in orginal_words_list:
                if orig_word in get_all_synonyms(word):
                    #print(word,":",get_all_synonyms(word))
                    modified_sus_paragraph.append((orig_word,pos))
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





def compute_word_similarity(filtered_sus_paragraph,filtered_ori_paragraph):
    orginal_words_list = create_original_wordlist(filtered_ori_paragraph)
    modified_sus_paragraph = modify_suspicious_paragraph(filtered_sus_paragraph,orginal_words_list)
    jac_similarity_mod = jaccard_similarity_modified(modified_sus_paragraph,filtered_ori_paragraph)
    result = round(jac_similarity_mod,3)
    return result