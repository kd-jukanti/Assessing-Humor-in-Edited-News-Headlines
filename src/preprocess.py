import re
import nltk
nltk.download('punkt')
from nltk import word_tokenize
import torch
def process_data(data, separate_lists=False):

    labels_list = data.meanGrade.to_list()
    o_headls_n_headls = []    
    new_word_list = []

    for (original, new_word) in zip(data.original.to_list(), data.edit.to_list()):
        # The dataset has 5 columns including the original headline and the single word edits, 
        # where the word to be replaced is included between (< and />).
        # Using the re python library, the original and the converted headlines are obtained.
        pattern = re.compile(r'\<(.*?)\/\>')
        origin_word = ''.join(re.findall(pattern, original))
        normal_origin_head = pattern.sub(origin_word, original)
        new_head = pattern.sub(new_word, original)

        o_headls_n_headls.append((normal_origin_head,new_head))
        new_word_list.append(new_word)
    
    if(separate_lists):
      o_headls = [i for i, j in o_headls_n_headls]
      n_headls = [j for i, j in o_headls_n_headls]
      return o_headls, n_headls, new_word_list, labels_list

    else:
      return o_headls_n_headls, labels_list, new_word_list


def get_tokenized_text(text):
  # NLTK word tokenizer is used to tokenize both the original headlines and the corresponding new edited headlines.
    tokenized_text = [] 
    for original, new in text:
      original = " ".join(word_tokenize(original))
      new = " ".join(word_tokenize(new))    

      tokenized_original = [token.lower() for token in original.split(' ')]
      tokenized_new = [token.lower() for token in new.split(' ')]

      tokenized_text.append((tokenized_original, tokenized_new))

    return tokenized_text


def get_word2idx(tokenized_text, new_word_list):
  # A dictionary is constructed for indexing the 
  # words in tokenized original headlines and the new words. 
  # An extra pad token is also included.
    vocabulary = []
    for original, new in tokenized_text:
      for token in original:
          if token not in vocabulary:
              vocabulary.append(token)
              
    for token in new_word_list:
      if token not in vocabulary:
          vocabulary.append(token)
  
    word2idx = {w: idx+1 for (idx, w) in enumerate(vocabulary)}
    word2idx['<pad>'] = 0
      
    return word2idx


def get_model_inputs(tokens, word2idx, labels):

  # Function get preproceed data as inputs for the data loader
  # The inputs are original headline tensor, edited headline tensor, and the label (score of humorness)

    vectorized = [([word2idx[tk] for tk in origin if tk in word2idx],[word2idx[tk] for tk in new if tk in word2idx]) for origin, new in tokens]

    original_lengths = [len(origin_headl) for origin_headl, new_headl in vectorized]
    new_lengths = [len(new_headl) for origin_headl, new_headl in vectorized]

    max_len = max(original_lengths)
    
    origin_tensor = torch.zeros((len(vectorized), max_len)).long()
    new_tensor = torch.zeros((len(vectorized), max_len)).long()

    for idx, ((origin_headl, new_headl), origin_headllen) in enumerate(zip(vectorized, original_lengths)):
      origin_tensor[idx, :origin_headllen] = torch.LongTensor(origin_headl)

    for idx, ((origin_headl, new_headl), new_headllen) in enumerate(zip(vectorized, new_lengths)):
      new_tensor[idx, :new_headllen] = torch.LongTensor(new_headl)  

    label_tensor = torch.FloatTensor(labels)
    
    return origin_tensor, new_tensor, label_tensor