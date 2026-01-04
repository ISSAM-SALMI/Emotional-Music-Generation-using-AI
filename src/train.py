import torch
import torch.nn as nn
import torch.optim as optim
from src import config, data_loader, model
import os
import time

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    train_loader = data_loader.get_dataloader(split='train')
    
    # Model
    transformer = model.MusicTransformer().to(device)
    
    optimizer = optim.Adam(transformer.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN)
    
    print("Starting training...")
    
    for epoch in range(config.EPOCHS):
        transformer.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            
            # Input: [BOS, EMOTION, ... tokens] (remove last)
            input_seq = batch[:, :-1]
            # Target: [EMOTION, ... tokens, EOS] (remove first)
            target_seq = batch[:, 1:]
            
            optimizer.zero_grad()
            output = transformer(input_seq)
            
            # Output: [batch, seq_len, vocab_size]
            # Target: [batch, seq_len]
            
            loss = criterion(output.reshape(-1, config.VOCAB_SIZE), target_seq.reshape(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), config.GRAD_CLIP)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{config.EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {elapsed:.2f}s | Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pt")
        torch.save(transformer.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    train()
