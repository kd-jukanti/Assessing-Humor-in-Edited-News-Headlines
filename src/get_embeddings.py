import torch
import torch.optim as optim
import torch.nn as nn
from model import ClassifyRNN
def set_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

# The pre-trained embedding is obtained by training a classification task 
# with the target as rounded scores. 
# These pre-trained embedding are used while training a Bidirectional RNN unit.
# This is a class to train that classifier and get the embeddings.
class train_classifier:
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, dropout, lr, dataloader, DEVICE):
        self.INPUT_DIM = input_dim
        self.EMBEDDING_DIM = embedding_dim
        self.HIDDEN_DIM = hidden_dim
        self.OUTPUT_DIM = output_dim
        self.DROPOUT = dropout

        self.LRATE = lr
        self.N_EPOCHS = 10

        self.model = ClassifyRNN(self.INPUT_DIM, 
                                        self.EMBEDDING_DIM, 
                                        self.HIDDEN_DIM, 
                                        self.OUTPUT_DIM,
                                        self.DROPOUT)
        self.model = self.model.to(DEVICE)
        self.datalooader = dataloader

    def train(self):
        set_seed(234)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.LRATE)
        steps = 36
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.N_EPOCHS):
            self.model.train()
            for original, new, labels in self.datalooader:       

                optimizer.zero_grad()
                predictions = self.model(new)

                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
        
        return self.model.embedding.weight.data
