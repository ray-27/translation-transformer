import pandas as pd
import re
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader


from torch.nn.utils.rnn import pad_sequence

class translation_dataset(Dataset):
    def __init__(self,dataframe,seq_len=100):

        english_sentences = dataframe['en_US'].tolist() 
        hindi_sentences = dataframe['hi_IN'].tolist()
        def tokenize(sentence):
            try:
                sen = sentence.split()
                return sen
            except:
                print("some exception found")


        english_tokens = [tokenize(sentence) for sentence in english_sentences]
        hindi_tokens = [tokenize(sentence) for sentence in hindi_sentences]
        # building a vocabulary

        english_vocab = {}
        english_vocab[0] = '<unk>'
        english_vocab[1] = '<pad>'
        # english_vocab[2] = '<end>'
        # english_vocab[3] = '<start>'

        hindi_vocab = {}
        hindi_vocab[0] = '<start>'
        hindi_vocab[1] = '<pad>'
        hindi_vocab[2] = '<end>'
        hindi_vocab[3] = '<unk>'
        i = 4
        def build_vocab(sentences,i,vocab):
            for sentence in sentences:
                for word in sentence:
                    word = re.sub(r'[,!@#$_;:`~]', '', word)
                    if word not in vocab:
                        if not word.isdigit() and word:
                            vocab[i] = word
                            i+=1
            return vocab
        english_vocab = build_vocab(english_tokens,2,english_vocab)
        hindi_vocab = build_vocab(hindi_tokens,4,hindi_vocab)

        # interchange the index and word in dict
        english_word_to_index = {word: index for index, word in english_vocab.items()}
        hindi_word_to_index = {word: index for index, word in hindi_vocab.items()}

        #convert sentences to sequences of indices
        def sentence_to_indices(sentence, word_to_index,eng):
            li = []
            if not eng:
                li = [word_to_index["<start>"]]
            for word in sentence.split(' '):
                # word = re.sub(r'[,!@#$_;:`~]', '', word)
                # sen += word + ' '
                li.append(word_to_index[word])
            if not eng:
                li.append(word_to_index["<end>"])
            
            return li
            # return [word_to_index[word] for word in sen]
        english_indices = [sentence_to_indices(sentence, english_word_to_index,eng=True) for sentence in english_sentences]
        hindi_indices = [sentence_to_indices(sentence, hindi_word_to_index,eng=False) for sentence in hindi_sentences]

        def pad_sequences(sequences, max_len=100, padding_value=1):
            if max_len is None:
                max_len = max(len(seq) for seq in sequences)
            return [seq + [padding_value] * (max_len - len(seq)) for seq in sequences]

        english_padded = pad_sequences(english_indices,max_len=seq_len)
        hindi_padded = pad_sequences(hindi_indices,max_len=seq_len)
        

        self.source_sequences = np.array(english_padded)
        self.target_sequences = np.array(hindi_padded)   

        self.english_vocab = english_vocab
        self.hindi_vocab = hindi_vocab

        self.english_vocab_len = len(english_vocab)
        self.hindi_vocab_len = len(hindi_vocab)

    def __len__(self):
        return len(self.source_sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.source_sequences[idx],dtype=torch.int32), torch.tensor(self.target_sequences[idx],dtype=torch.int32)

        
if __name__ == "__main__":
    # splits = {'train': 'train.csv', 'validation': 'validation.csv', 'test': 'test.csv'}
    # data = pd.read_csv("hf://datasets/Amani27/massive_translation_dataset/" + splits["train"])
    data = pd.read_csv('data/hugging_data.csv')
    dataset = translation_dataset(data)
    
    english_vocab = dataset.english_vocab
    hindi_vocab = dataset.hindi_vocab
    eng_voc_len = dataset.english_vocab_len
    hin_voc_len = dataset.hindi_vocab_len
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
   
    # print(type(english_vocab))
    # print(english_vocab[20])
    for x,y in dataset:
        print(x.shape)
        print(y.shape)
        break
    print(x)
    print(y)

    for i in x.numpy():
        word = english_vocab[i] 
        if word != '<pad>':
            print(word+" ")
    
    for a in y.numpy():
        word = hindi_vocab[a] 
        if word != '<pad>':
            print(word+" ")

    print(eng_voc_len,hin_voc_len)
    print(hindi_vocab[64822])