import yaml, random, time, os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from model import FFNN, RNN
from get_embeddings import train_classifier
from dataloader import get_dataloader, get_dataloader_classification

if torch.cuda.is_available():
  torch.backends.cudnn.deterministic = True
  DEVICE='cuda:0'
else:
  DEVICE='cpu'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)   

# Function to write predictions to output file
def write_predictions(predictions, test_data_frame, out_loc):
    test_data_frame['pred'] = predictions
    output = test_data_frame[['id','pred']]
    output.to_csv(out_loc, index=False)
        
    print('Output file created:\n\t- '+os.path.abspath(out_loc))

# function to get RMSE score based on ground truth and predictions
def score(truth_loc, prediction_loc):
    truth = pd.read_csv(truth_loc, usecols=['id','meanGrade'])
    pred = pd.read_csv(prediction_loc, usecols=['id','pred'])
    
    assert(sorted(truth.id) == sorted(pred.id)),"ID mismatch between ground truth and prediction!"
    
    data = pd.merge(truth,pred)
    rmse = np.sqrt(np.mean((data['meanGrade'] - data['pred'])**2))
    
    print("RMSE = %.6f" % rmse)

# Function for evaluation.
# Get predictions in eval() mode and return the loss
def evaluate(model, valid_dataloader):
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for original, new, labels in valid_dataloader:

            predictions = model(original, new).squeeze(1)
            loss = torch.sqrt(((predictions - labels)**2).mean())
            epoch_loss += loss.item()

    return epoch_loss / len(valid_dataloader)


def main(params, train_dataloader, valid_dataloader, optimizer, scheduler, model):

    for epoch in range(params['hyperparameters']['epochs']):
        start_time = time.time()
        # model training
        model.train()        
        epoch_loss = 0

        for original, new, labels in train_dataloader:
            # Zero the gradients
            optimizer.zero_grad()
            # Get predictions after forward pass of the model
            predictions = model(original, new)
            # compute the loss
            loss = torch.sqrt(((predictions - labels)**2).mean())
            
            # Gradients are calculated for each parameter
            # Parameters are updated using the gradients and optimizer algorithm
            # Learning rate is updated
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        average_epoch_loss = epoch_loss / len(train_dataloader)
        
        end_time = time.time()

        average_epoch_valid_loss = evaluate(model, valid_dataloader)

        print(f'Epoch: {epoch+1:02} | Epoch Time: {end_time-start_time}')
        print(f'\tTrain Loss: {average_epoch_loss:.3f} | Val. Loss: {average_epoch_valid_loss:.3f} ')

    return model

def test_predictions(model, test_dataloader):
    test_loss = 0
    test_predictions = []

    model.eval()

    with torch.no_grad():

        for original, new, labels in test_dataloader:
            
            predictions__batch = model(original, new).squeeze(1)
            test_predictions += predictions__batch.tolist()
            loss = torch.sqrt(((predictions__batch - labels)**2).mean())
            test_loss += loss.item()

        average_test_loss = test_loss / len(test_dataloader)

    print(f'| Test Loss: {average_test_loss:.6f} |')

    out_loc = 'task-1-output.csv'
    test = pd.read_csv(params['dataset']['data_dir']+'test.csv')
    write_predictions(test_predictions, test, out_loc)

    truth_loc = params['dataset']['data_dir']+'test.csv'
    prediction_loc = 'task-1-output.csv'
    score(truth_loc, prediction_loc)



if __name__=="__main__":
    set_seed(234)
    # the arguments are given from a config file
    with open("/content/drive/MyDrive/Funniness_estimation/config.yaml") as file:
        params = yaml.safe_load(file)

    # Get the dataloaders
    train_dataloader, valid_dataloader, test_dataloader, word2idx = get_dataloader(params, DEVICE)
    # Construct the model and optimizer
    if(params['model']['name'] == 'NN'):
        
        model = FFNN(params['hyperparameters']['embedding_dim_NN'],
                            params['hyperparameters']['hidden_dim_1'],
                            params['hyperparameters']['hidden_dim_2'],
                            params['hyperparameters']['hidden_dim_3'],
                            len(word2idx))

        model = model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=params['hyperparameters']['lr'])
    
    if(params['model']['name'] == 'RNN'):
        # Get the dataloaders
        clf_train_dataloader, clf_valid_dataloader = get_dataloader_classification(params, DEVICE)
        # Get the embeddings
        classifier = train_classifier(len(word2idx), 50, 128, 4, 0.4, 1e-4, clf_train_dataloader,DEVICE)
        ebd = classifier.train()
        # define model
        model = RNN(params['hyperparameters']['embedding_dim_RNN'], 
                            params['hyperparameters']['hidden_dim'],
                            params['hyperparameters']['fc_output_dim'],
                            params['hyperparameters']['dropout_RNN'],
                            ebd)
        model = model.to(DEVICE)
        # define optimizer
        optimizer = optim.AdamW(model.parameters(), lr=params['hyperparameters']['lr'])

    print(model)
    
    # define scheduler for learning rate
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, params['hyperparameters']['steps'])
    # Train and get the model
    model = main(params, train_dataloader, valid_dataloader, optimizer, scheduler, model)
    # Final evaluation on test dataset
    test_predictions(model, test_dataloader)



