import torch
import torch.nn as nn
import math
from src import config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MusicTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(config.VOCAB_SIZE, config.EMBED_DIM, padding_idx=config.PAD_TOKEN)
        self.pos_encoder = PositionalEncoding(config.EMBED_DIM, config.SEQ_LEN)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config.EMBED_DIM,
            nhead=config.N_HEADS,
            dim_feedforward=config.FF_DIM,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=config.N_LAYERS)
        
        self.output_head = nn.Linear(config.EMBED_DIM, config.VOCAB_SIZE)
        
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_head.bias.data.zero_()
        self.output_head.weight.data.uniform_(-initrange, initrange)
        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        # src: [batch_size, seq_len]
        mask = self._generate_square_subsequent_mask(src.size(1)).to(src.device)
        
        # Create padding mask
        # src_key_padding_mask: [batch_size, seq_len]
        # True where value is PAD_TOKEN
        # We use a float mask to match the type of 'mask' (causal mask) to avoid warnings
        src_key_padding_mask = torch.zeros(src.size(), device=src.device)
        src_key_padding_mask = src_key_padding_mask.masked_fill(src == config.PAD_TOKEN, float('-inf'))
        
        x = self.embedding(src) * math.sqrt(config.EMBED_DIM)
        x = self.pos_encoder(x)
        
        output = self.transformer_encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask)
        output = self.output_head(output)
        
        return output
