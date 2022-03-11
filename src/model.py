import torch
import torch.nn as nn

class FFNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, vocab_size):  
        super(FFNN, self).__init__()
        # embedding layer
        # The padding idx option ensures that the 0-th token in the vocabulary is utilised for padding.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.fc1 = nn.Linear(embedding_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.relu1 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)


    def forward(self, x, y):
        # put x into embedding layer
        x_embedded = self.embedding(x)
        # Compute the average embeddings that ignores padding
        x_headl_lens = x.ne(0).sum(1, keepdims=True)
        x_averaged = x_embedded.sum(1) / x_headl_lens

        # Get averaged embedding for y also
        y_embedded = self.embedding(y)       
        y_headl_lens = y.ne(0).sum(1, keepdims=True)
        y_averaged = y_embedded.sum(1) / y_headl_lens

        # FC layers for x and y along with activation layer
        x_out = self.fc1(x_averaged)
        y_out = self.fc1(y_averaged)

        x_out = self.relu1(x_out)
        y_out = self.relu1(y_out)

        x_out = self.fc2(x_out)
        y_out = self.fc2(y_out)

        x_out = self.relu1(x_out)
        y_out = self.relu1(y_out)


        x_out = self.fc3(x_out)
        y_out = self.fc3(y_out)


        out = x_out * y_out 
        out = torch.sum(out, 1, keepdim = True)

        return out


class RNN(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, fc_out_dim,
                 dropout, embeddings):

        super().__init__()

        self.hidden_dim = hidden_dim
        # Fine-tune the pre-trained embeddings
        self.embedding = nn.Embedding.from_pretrained(embeddings, padding_idx=0)
        # Bidirectional RNN module
        self.rnn = nn.RNN(embedding_dim,
                          hidden_dim,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=1)
        
        # The final hidden state is passed through a fully connected layer by the linear layer.
        # Bidirectional RNN, following are concatenated:
            #  - The last hidden state from the forward RNN
            #  - The last hidden state from the backward RNN
          # Thus hidden size is doubled
        linear_hidden_in = hidden_dim * 2
        self.fc = nn.Linear(linear_hidden_in, fc_out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text1, text2):

        embedded1 = self.dropout(self.embedding(text1))
        embedded2 = self.dropout(self.embedding(text2))

        # An RNN in PyTorch returns two values:
        # (1) All hidden states of the last RNN layer
        # (2) Hidden state of the last timestep for every layer
        all_hidden1, last_hidden1 = self.rnn(embedded1)
        all_hidden2, last_hidden2 = self.rnn(embedded2)

        # Concat the final forward (hidden[0,:,:]) and backward (hidden[1,:,:]) hidden layers
        last_hidden1 = torch.cat((last_hidden1[0, :, :], last_hidden1[1, :, :]), dim=-1)
        last_hidden2 = torch.cat((last_hidden2[0, :, :], last_hidden2[1, :, :]), dim=-1)


        out1 = self.fc(self.dropout(last_hidden1))
        out2 = self.fc(self.dropout(last_hidden2))
        # Final predictions
        out = out1 * out2 
        preds = torch.sum(out, 1, keepdim = True)   
        
        return preds

class ClassifyRNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 dropout):

        super().__init__()

        self.hidden_dim = hidden_dim       

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) 
        
        self.rnn = nn.RNN(embedding_dim,
                          hidden_dim,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=1)
      

        linear_hidden_in = hidden_dim * 2
        self.fc = nn.Linear(linear_hidden_in, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):

        embedded = self.dropout(self.embedding(text))
        
        all_hidden, last_hidden = self.rnn(embedded)
        last_hidden = torch.cat((last_hidden[0, :, :], last_hidden[1, :, :]), dim=-1)

        logits = self.fc(self.dropout(last_hidden))
          
        return logits