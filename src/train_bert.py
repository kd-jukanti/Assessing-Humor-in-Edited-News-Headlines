import yaml, random, time, os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from dataloader import get_dataloader_bert
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup

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
        for input_ids_batch, attention_mask_batch, token_type_ids_batch, labels in valid_dataloader:

            outputs = model(input_ids_batch,
                            attention_mask=attention_mask_batch,
                            token_type_ids=token_type_ids_batch)
            predictions = outputs[0].squeeze(1)
            loss = torch.sqrt(((predictions - labels)**2).mean())
            epoch_loss += loss.item()

    return epoch_loss / len(valid_dataloader)


def main(params, train_dataloader, valid_dataloader, optimizer, scheduler, model):

    for epoch in range(params['hyperparameters']['epochs']):
        start_time = time.time()
        # model training
        model.train()        
        epoch_loss = 0

        for input_ids_batch, attention_mask_batch, token_type_ids_batch, labels in train_dataloader:
            # Zero the gradients
            optimizer.zero_grad()
            # Get predictions after forward pass of the model
            outputs = model(input_ids_batch,
                            attention_mask=attention_mask_batch,
                            token_type_ids=token_type_ids_batch)
            predictions = outputs[0].squeeze(1)
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

        for input_ids_batch, attention_mask_batch, token_type_ids_batch, labels in test_dataloader:
            
            predictions__batch = model(input_ids_batch,
                           attention_mask=attention_mask_batch,
                           token_type_ids=token_type_ids_batch)[0].squeeze(1)
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
    with open("/content/drive/MyDrive/Funniness_estimation/bert_config.yaml") as file:
        params = yaml.safe_load(file)

    # Bert tokenizer
    tokenizer = BertTokenizer.from_pretrained(params['model']['name'],do_lower_case=True)
    # dataloader for bert model
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader_bert(params, DEVICE, tokenizer)
    # Construct the model
    # Load the BertForSequenceClassification model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                        num_labels = 1,   
                                                        output_attentions = False,
                                                        output_hidden_states = False)

    TOTSTEPS = len(train_dataloader) * params['hyperparameters']['epochs'] * 2
    WUSTEPS = int(TOTSTEPS * float(params['hyperparameters']['WU']))
    # Apply weight decay to all parameters other than bias and layer normalization terms
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if "bert" not in n], 'lr': float(params['hyperparameters']['LRATE']), 'weight_decay': float(params['hyperparameters']['WDECAY'])},
        {'params': [p for n, p in model.named_parameters() if "bert" in n], 'weight_decay':  float(params['hyperparameters']['WDECAY'])}
    ]
    # define optimizer and scheduler
    optimizer = AdamW(optimizer_grouped_parameters, lr=float(params['hyperparameters']['FRATE']), eps = float(params['hyperparameters']['EPS']))
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = WUSTEPS,
                                            num_training_steps = TOTSTEPS)

    model = model.to(DEVICE)
    model = main(params, train_dataloader, valid_dataloader, optimizer, scheduler, model)
    test_predictions(model, test_dataloader)




