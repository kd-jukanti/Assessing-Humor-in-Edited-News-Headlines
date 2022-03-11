from torchtext import data
from torchtext import datasets
import torch.utils.data as tud
from get_data import read_data

# Dataset class for data for FFNN and RNN
class headlinesDataset(tud.Dataset):
    def __init__(self, X, DEVICE, classify = False, y = None):
        self.len = X[0].shape[0]

        self.x1_data = X[0].to(DEVICE)
        self.x2_data = X[1].to(DEVICE)
        if classify:
            self.y1_data = y.to(DEVICE)
        else:
            self.y1_data = X[2].to(DEVICE)

    def __getitem__(self, index):
        return self.x1_data[index], self.x2_data[index], self.y1_data[index]


    def __len__(self):
        return self.len
# Dataset class for data for BERT
class BERT_Dataset(tud.Dataset):
    def __init__(self, X, DEVICE):
        self.len = X[0].shape[0]

        self.x1_data = X[0].to(DEVICE)
        self.x2_data = X[1].to(DEVICE)
        self.x3_data = X[2].to(DEVICE)
        self.y1_data = X[3].to(DEVICE)


    def __getitem__(self, index):
        return self.x1_data[index], self.x2_data[index], self.x3_data[index], self.y1_data[index]


    def __len__(self):
        return self.len

def get_dataloader(params, DEVICE):
    # The dataloader for the model is constructed with the vectorized original headlines, 
    # edited headlines and the labels(the mean score of funniness).
    # This dataloader for FFNN and RNN model

    BATCH_SIZE = params['hyperparameters']['batch_size']
    val_batch_size = params['hyperparameters']['val_batch_size']
    data = read_data(params)
    train, valid, test = data.preprocess_data()
    train_dataset = headlinesDataset(train, DEVICE)
    valid_dataset = headlinesDataset(valid, DEVICE)
    test_dataset = headlinesDataset(test, DEVICE)

    train_dataloader = tud.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = tud.DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=True)
    test_dataloader = tud.DataLoader(test_dataset, batch_size=val_batch_size, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader, data.word2idx

def get_dataloader_classification(params, DEVICE):
    # The dataloader for the model is constructed with the vectorized original headlines, 
    # edited headlines and the labels(the mean funniness scores are rounded to nearest integers).
    # This dataloader for training a classification RNN model for training embeddings.

    BATCH_SIZE = params['hyperparameters']['batch_size']
    data = read_data(params)
    train, valid, test = data.preprocess_data()


    classify_train_dataset = headlinesDataset(train, DEVICE, True, data.get_classification_labels(train[2]))
    classify_valid_dataset = headlinesDataset(valid, DEVICE, True, data.get_classification_labels(valid[2]))

    classify_train_dataloader = tud.DataLoader(classify_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    classify_valid_dataloader = tud.DataLoader(classify_valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return classify_train_dataloader, classify_valid_dataloader

def get_dataloader_bert(params, DEVICE, tokenizer):
    # The dataloader for the model is constructed with the input_ids, attention_mask, token_type ids, labels
    # This dataloader for training a classification BERT.

    BATCH_SIZE = params['hyperparameters']['batch_size']
    val_batch_size = params['hyperparameters']['val_batch_size']
    data = read_data(params)
    train, valid, test = data.preprocess_bert_data(tokenizer)
    train_dataset = BERT_Dataset(train, DEVICE)
    valid_dataset = BERT_Dataset(valid, DEVICE)
    test_dataset = BERT_Dataset(test, DEVICE)

    train_dataloader = tud.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = tud.DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=True)
    test_dataloader = tud.DataLoader(test_dataset, batch_size=val_batch_size, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader