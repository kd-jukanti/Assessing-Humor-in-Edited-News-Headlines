import torch
import pandas as pd
from preprocess import process_data, get_tokenized_text, get_word2idx, get_model_inputs

class read_data:
    def __init__(self, params):
        # read data from .csv
        self.train = pd.read_csv(params['dataset']['data_dir']+'train.csv')
        self.test = pd.read_csv(params['dataset']['data_dir']+'test.csv')
        self.valid = pd.read_csv(params['dataset']['data_dir']+'dev.csv')
        self.word2idx = None
    
    # Function to preprocess data for training, validation and test.
    # Data for FFNN and RNN 
    # The preprocessing functions are defined in preprocess.py
    # Steps are : get headlines, tokenize using word tokenizer, build vocabulary, get the final inputs(original, edited, labels)
    def preprocess_data(self):
        text, labels, new_word_list = process_data(self.train)
        tokenized_text = get_tokenized_text(text)
        self.word2idx = get_word2idx(tokenized_text, new_word_list)
        train_original, train_new, train_labels = get_model_inputs(tokenized_text, self.word2idx, labels)

        text, labels, new_word_list = process_data(self.valid)
        tokenized_text = get_tokenized_text(text)
        word2idx = get_word2idx(tokenized_text, new_word_list)
        valid_original, valid_new, valid_labels = get_model_inputs(tokenized_text, word2idx, labels)

        text, labels, new_word_list = process_data(self.test)
        tokenized_text = get_tokenized_text(text)
        word2idx = get_word2idx(tokenized_text, new_word_list)
        test_original, test_new, test_labels = get_model_inputs(tokenized_text, word2idx, labels)

        return [train_original, train_new, train_labels], [valid_original, valid_new, valid_labels], [test_original, test_new, test_labels]

    def get_classification_labels(self, labels):
        return torch.round(labels).long()


    # Function to preprocess data for training, validation and test.
    # Data for BERT
    # The preprocessing functions are defined in preprocess.py
    # Steps are : get headlines, tokenize using BERT tokenizer, 
    # get the final inputs(input_id, attention_mask, token_type_ids, labels) from the encoded inputs after bert tokenizer
    def preprocess_bert_data(self, tokenizer):
        train_o_headls, train_n_headls, train_new_word_list, train_labels_list = process_data(self.train, True)
        valid_o_headls, valid_n_headls, valid_new_word_list, valid_labels_list = process_data(self.valid, True)
        test_o_headls, test_n_headls, test_new_word_list, test_labels_list = process_data(self.test, True)

        train_encoded_inputs = tokenizer(train_n_headls, train_new_word_list, padding='max_length', max_length=38, truncation=True, return_tensors="pt")
        valid_encoded_inputs = tokenizer(valid_n_headls, valid_new_word_list, padding='max_length', max_length=38, truncation=True, return_tensors="pt")
        test_encoded_inputs = tokenizer(test_n_headls, test_new_word_list, padding='max_length', max_length=38, truncation=True, return_tensors="pt")

        train_input_ids = train_encoded_inputs['input_ids']
        train_attention_mask = train_encoded_inputs['attention_mask']
        train_token_type_ids = train_encoded_inputs['token_type_ids']
        train_labels = torch.tensor(train_labels_list)

        valid_input_ids = valid_encoded_inputs['input_ids']
        valid_attention_mask = valid_encoded_inputs['attention_mask']
        valid_token_type_ids = valid_encoded_inputs['token_type_ids']
        valid_labels = torch.tensor(valid_labels_list)

        test_input_ids = test_encoded_inputs['input_ids']
        test_attention_mask = test_encoded_inputs['attention_mask']
        test_token_type_ids = test_encoded_inputs['token_type_ids']
        test_labels = torch.tensor(test_labels_list)

        return [train_input_ids, train_attention_mask, train_token_type_ids, train_labels], [valid_input_ids, valid_attention_mask, valid_token_type_ids, valid_labels],[test_input_ids, test_attention_mask, test_token_type_ids, test_labels]
