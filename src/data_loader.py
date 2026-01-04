import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from tqdm import tqdm
from src import config, midi_processor

class EmopiaDataset(Dataset):
    def __init__(self, split='train'):
        self.data = []
        
        # Load labels
        if not os.path.exists(config.LABEL_FILE):
            print(f"Warning: Label file not found at {config.LABEL_FILE}")
            return

        df = pd.read_csv(config.LABEL_FILE)
        
        # Shuffle and split
        # We use a fixed seed for reproducibility
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        split_idx = int(len(df) * 0.9)
        if split == 'train':
            df = df.iloc[:split_idx]
        else:
            df = df.iloc[split_idx:]
            
        print(f"Loading {split} dataset ({len(df)} samples)...")
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            mid_id = row['ID']
            label_str = row['4Q']
            
            # Map to Q1..Q4
            emotion_key = f"Q{label_str}"
            if emotion_key not in config.EMOTION_TO_ID:
                continue
            emotion_id = config.EMOTION_TO_ID[emotion_key]
            
            # Construct file path
            # Try exact match
            file_path = os.path.join(config.DATA_DIR, f"{mid_id}.mid")
            if not os.path.exists(file_path):
                continue
                
            tokens = midi_processor.encode_midi(file_path)
            if tokens is None or len(tokens) == 0:
                continue
                
            # Truncate if too long (leave room for BOS, EMOTION, EOS)
            max_len = config.SEQ_LEN - 3
            tokens = tokens[:max_len]
            
            self.data.append({
                'tokens': tokens,
                'emotion': emotion_id
            })
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    # batch is list of dicts
    batch_input_ids = []
    
    for item in batch:
        tokens = item['tokens']
        emotion_id = item['emotion']
        
        # Construct sequence: [BOS, EMOTION, ... tokens, EOS]
        emotion_token = config.TOKEN_OFFSET_EMOTION + emotion_id
        
        seq = [config.BOS_TOKEN, emotion_token] + tokens + [config.EOS_TOKEN]
        batch_input_ids.append(torch.tensor(seq, dtype=torch.long))
        
    # Pad
    padded_input = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=config.PAD_TOKEN)
    
    return padded_input

def get_dataloader(split='train', batch_size=config.BATCH_SIZE):
    dataset = EmopiaDataset(split=split)
    shuffle = (split == 'train')
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
