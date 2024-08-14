import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import translation_dataset
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import math

class gpt_TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_size=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1):
        super(gpt_TransformerModel, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout
        )

        self.output_layer = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, src, trg):
        src = self.positional_encoding(self.src_embedding(src))
        trg = self.positional_encoding(self.tgt_embedding(trg))

        print(f'src shape : {src.shape}, trg shape : {trg.shape}')
        # src and trg masks can be generated if necessary, especially for padding tokens
        output = self.transformer(src, trg)
        return self.output_layer(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on
        # pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class translation_model(nn.Module):
    def __init__(self,src_vocab_size, tgt_vocab_size, emb_dim=64, nhead=8, nhid=512, nlayers=6, max_len=100,dropout=0.5):
        super().__init__()
        self.emb_dim=emb_dim
        self.nhead=nhead
        self.nhid=nhid
        self.nlayers=nlayers
        self.seq_len=max_len
        self.dropout=dropout

        
        self.src_embedding = nn.Embedding(src_vocab_size, emb_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, emb_dim)

        # Positional encoding (shared for simplicity, can be separated if needed)
        self.pos_encoder = nn.Embedding(max_len, emb_dim) 

        # Transformer model
        self.transformer = nn.Transformer(emb_dim, nhead, nlayers, nlayers, nhid, dropout)
        
        # Output layer specific to the target language (Hindi)
        self.fc_out = nn.Linear(emb_dim, tgt_vocab_size)

        self.src_pad_idx = 1
    
    def forward(self, src, tgt):
        # Positional encoding
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]
        src_positions = torch.arange(0, src_seq_len).unsqueeze(0).expand(src.shape[0], src_seq_len).to(src.device)
        tgt_positions = torch.arange(0, tgt_seq_len).unsqueeze(0).expand(tgt.shape[0], tgt_seq_len).to(tgt.device)

        # Embedding the source and target sequences
        src = self.src_embedding(src) + self.pos_encoder(src_positions)
        tgt = self.tgt_embedding(tgt) + self.pos_encoder(tgt_positions)

        # Transformer forward pass
        # print(f"model : src shape : {src.shape}, trg shape : {tgt.shape}")
        output = self.transformer(src, tgt)
        
        # Output layer for target language
        output = self.fc_out(output)

        return output

def predict_next_token(model, src, tgt_input):
    model.eval()
    output_logits = model(src, tgt_input)
    next_token_probs = F.softmax(output_logits[:, -1, :], dim=-1)
    next_token = torch.argmax(next_token_probs, dim=-1, keepdim=True)
    return next_token

if __name__ == "__main__":

    data = pd.read_csv('data/hugging_data.csv')
    dataset = gpt_data(data)
    loader = DataLoader(dataset)
    src_vocab_size = dataset.english_vocab_len  # English vocabulary size
    tgt_vocab_size = dataset.hindi_vocab_len # Hindi vocabulary size


    model = translation_model(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            emb_dim=512, 
            nhead=8, 
            nhid=512, 
            nlayers=6, 
            dropout=0.1
    )

    # for x,y in loader:
    #     break
    src = torch.randint(0, 79736, (32, 10))  # Random example data [batch_size, src_seq_len]
    trg = torch.randint(0, 87428, (32, 12))  # Random example data [batch_size, trg_seq_len] including <start> and <end>

    # Prepare trg_input (excluding <end> token) and trg_labels (excluding <start> token)
    trg_input = trg[:, :-1]
    trg_labels = trg[:, 1:]

    # Training loop
    output = model(src, trg_input)

    print(output.shape)
    # out = model(x,y)
    # print(out.shape)
    # next_token = predict_next_token(model, x, y)
    # print(f"Predicted next token ID: {next_token.item()}")

    # Append the predicted token to tgt_input for the next step
    tgt_input = torch.cat((tgt_input, next_token), dim=1)
    print(f"Updated tgt_input: {tgt_input}")