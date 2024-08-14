import json
import pandas as pd
import torch
import torch.nn.functional as F
from colorama import Fore,init
from dataset import translation_dataset
import config
from tqdm import tqdm

init(autoreset=True)


# eng_vocab_path = "model_loads/eng_vocab.json"
# hin_vocab_path = "model_loads/hin_vocab.json"
# # Reading JSON data
 # Load data from JSON file into a dictionary

def device_is():
    if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Device is " + Fore.GREEN + "CUDA")
    else:
        device = torch.device('mps')
        print("Device is "+Fore.YELLOW+  "CPU")
    
    return device

def tensor_to_sentence(x,vocab):
    if type(x) == torch.Tensor:
        x = x.tolist()
    
    for i in x:
        word = vocab[f'{i}']
        if word == "<start>":
            continue
        elif word == "<end>":
            break
        else:
            print(word)

def sentence_to_tensor(x,vocab): #-> a string, like = "Hi how are you ?"
    
    s = x.lower().split(' ')
    # converting it to index from vocab
    eng_index = []
    for word in s:
        if f'{word}' not in vocab:
            index = int(vocab['<unk>'])
        else:
            index = int(vocab[f'{word}'])
        eng_index.append(index)
    
    return eng_index

def pad_sequences(sequences, max_len=100, padding_value=1):
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    return [seq + [padding_value] * (max_len - len(seq)) for seq in sequences]

def model_to_index(eng,model,device='cpu',prin=False):
    # print(type(eng))
    eng = torch.tensor(eng).to(torch.int32).unsqueeze(0) # -> [1,seq_len]
    
    batch, seq_len = eng.shape #[1,seq_len]
    hin = torch.ones(eng.shape).to(torch.int32)
    hin[0][0] = 0 #this is the index of <start> token

    translated_hin = []
    # both have shape [1,seq_len]
    # print("hindi satarting : ",eng)
    # print("hindi satarting : ",hin)
    hin = hin.to(device)
    eng = eng.to(device)

    if prin:
        print(Fore.BLUE + "traslating....")
    
    for i in range(1,seq_len):
        out = model(eng,hin)
        probabilities = F.softmax(out, dim=2)
        most_probable_indices = torch.argmax(probabilities[0, 0, :], dim=0)

        prob_index = most_probable_indices.item()
        hin[0][i] = prob_index

        if prob_index == 2:
            break
    
    return hin.to('cpu')
        # print(most_probable_indices) 
        # out_dim = out.shape[-1]
        # out = out.contiguous().view(-1,out_dim)

def translate(x,model,seq_len,inverted_eng_dict,hin_vocab):
    s = sentence_to_tensor(x,inverted_eng_dict)
    s = pad_sequences([s],max_len=seq_len)
    s = s[0]
    hin = model_to_index(s,model)
    lis = hin[0].tolist()
    hindi = ""
    for i in lis:
        word = hin_vocab[f'{i}']
        # print(word)
        if word != '<start>' and word != '<pad>':
            hindi += " " + word
        elif word == '<end>':
            break
    return hindi

def val(mod):
    eng_vocab_path = "model_loads/eng_vocab.json"
    hin_vocab_path = "model_loads/hin_vocab.json"

    with open(eng_vocab_path, 'r') as json_file:
        eng_vocab = json.load(json_file)  # Load data from JSON file into a dictionary

    with open(hin_vocab_path, 'r') as json_file:
        hin_vocab = json.load(json_file) 
    
    with open(eng_vocab_path,'r') as json_file:
        eng_index = json.load(json_file)
        inverted_eng_dict = {value: key for key, value in eng_index.items()}
    
    # model = torch.load(f'model_loads/translation_model_{config.model_subname}.pth').to('cpu')


    seq_len = config.seq_len

    x = "wake me up at nine am on friday"
    model = mod
    hindi = translate(x,model,seq_len,inverted_eng_dict,hin_vocab)


    return hindi

if __name__ == "__main__":
    # with open(eng_vocab_path, 'r') as json_file:
    #     eng_vocab = json.load(json_file)  # Load data from JSON file into a dictionary

    # with open(hin_vocab_path, 'r') as json_file:
    #     hin_vocab = json.load(json_file) 
    
    # with open(eng_vocab_path,'r') as json_file:
    #     eng_index = json.load(json_file)
    #     inverted_eng_dict = {value: key for key, value in eng_index.items()}

    # device = device_is()
    # #load the model
    # model = torch.load('model_loads/translation_model.pth').to('cpu')
    
    # # check the seq_len
    # seq_len = config.seq_len

    # x = "How are you bbbb"

    # hindi = translate(x,model,seq_len,inverted_eng_dict)
    # print(hindi)
    # # if len(x) > seq_len:
    # #     print(Fore.RED + "Sentence excided the contex window")
    # #     continue
    # s = sentence_to_tensor(x,inverted_eng_dict)

    device = device_is()
    model = torch.load(f'model_loads/translation_model_{config.model_subname}.pth').to(device)
    print(val(model))
