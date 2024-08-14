import pandas as pd
import torch
from torch import nn, optim
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from model import translation_model, gpt_TransformerModel
from dataset import translation_dataset
from tqdm import tqdm
import config
import json
from colorama import init,Fore
init(autoreset=True)

#load the config values
seq_len = config.seq_len
batch_size = config.batch_size
num_workers = config.num_workers
shuffle = config.shuffle
pin_memory = config.pin_memory
emb_dim=config.emb_dim
nhead=config.nhead
nhid=config.nhid
nlayers=config.nlayers
dropout=config.dropout
lr=config.learning_rate
epoches = config.epoches

def device_is():
    if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Device is " + Fore.GREEN + "CUDA")
    else:
        device = torch.device('mps')
        print("Device is "+Fore.YELLOW+  "CPU")
    
    return device

def batch_conv(hin,i):
    batch,seq_len = hin.shape # -> [batch,100]
    # we iterate over the batch, take each tensor copy them till the i'th  index
    # all else values will be padded i.e '1' 
    # and if we reach the <end> token, then we don't change the tensor we just push it as it is
    # print(temp_tensor)
    tensor_li = []
    for tensor in hin: 
        temp_tensor = torch.ones(seq_len).to(torch.int32)
        #tensor -> [100]
        for j in range(i+1):
            if tensor[j] == 2:
                temp_tensor[j] = tensor[j]
                break
            else:
                temp_tensor[j] = tensor[j]
        tensor_li.append(temp_tensor)
    
    return torch.stack(tensor_li,dim=0)

def train():
    device = device_is()
    # loading the data 
    data = pd.read_csv('data/hugging_data.csv',nrows=100)
    # train_df,val_df = train_test_split(data,test_size=0.1)
    dataset = translation_dataset(data,seq_len=seq_len)
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=shuffle,
                                        pin_memory=pin_memory)

    src_vocab_size = dataset.english_vocab_len
    tgt_vocab_size = dataset.hindi_vocab_len

    english_vocab = dataset.english_vocab
    hindi_vocab = dataset.hindi_vocab
    #saving the vocab to as json for future inference 
    eng_vocab_path = "model_loads/eng_vocab.json"
    hin_vocab_path = "model_loads/hin_vocab.json"
    with open(eng_vocab_path, 'w') as json_file:
        json.dump(english_vocab, json_file, indent=4)
    with open(hin_vocab_path, 'w') as json_file:
        json.dump(hindi_vocab, json_file, indent=4)


    #load this model
    model = translation_model(
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size,
                emb_dim=emb_dim, 
                nhead=nhead, 
                nhid=nhid, 
                nlayers=nlayers, 
                dropout=dropout,
                max_len=seq_len
        ).to(device)
    # model = gpt_TransformerModel(
    #      src_vocab_size=src_vocab_size,
    #      tgt_vocab_size=tgt_vocab_size,
    #      emb_size=emb_dim,
    #      nhead=nhead,
    # )


    optimizer = Adam(params=model.parameters(),
                    lr=lr)

    # # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
    # #                                                  verbose=True,
    # #                                                  factor=factor,
    # #                                                  patience=patience)

    criterion = nn.CrossEntropyLoss(ignore_index=1) #ignore index = 1 means it will ignore the value 1 in the calculation as it is just the index for padding 

    l = len(loader)
    loss_lis = []
    for epoch in tqdm(range(epoches), desc="epoch loop"):
        model.train()
        epoch_loss = 0
        
        for ba, (eng, hin) in enumerate(loader):
            batch_loss = 0
            eng = eng.to(device)
            hin = hin.to(device)

            # hin -> [batch,100,512]
            # print(eng.shape)
            # print(hin.shape)
            seq_loss_lis = []
            for i in tqdm(range(seq_len), desc="sequence loop"):
                converted_batch = batch_conv(hin,i)
                converted_batch = converted_batch.to(device)
                optimizer.zero_grad()

                output = model(eng, converted_batch)
            

                output_dim = output.shape[-1] # gives the last value in the shape
                output = output.contiguous().view(-1, output_dim)
                trg_labels = hin.contiguous().view(-1).to(torch.long)

                loss = criterion(output, trg_labels)
                seq_loss = loss.item()

                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
        
                tqdm.write(f"Batch: {ba+1}, sequence len: {i}, Loss: {seq_loss}")
                seq_loss_lis.append(seq_loss)
        
            batch_loss += sum(seq_loss_lis)/seq_len
            tqdm.write(f"batch loss of {ba+1} is {batch_loss}")
            epoch_loss += batch_loss
        ## valadition 

        tqdm.write(f"Epoch {epoch} finished with average loss: {epoch_loss/l}")
        loss_lis.append(epoch_loss / l)
    
    print(Fore.GREEN + "Training complete")
    #saving the model
    torch.save(model,'model_loads/translation_model.pth')
    print(Fore.GREEN + "Model saved")

if __name__ == "__main__":
     train()