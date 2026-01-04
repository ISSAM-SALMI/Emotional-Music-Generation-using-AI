import torch
import torch.nn.functional as F
from src import config, model, midi_processor
import os
import argparse
# from midi2audio import FluidSynth # Optional dependency

def generate(emotion, checkpoint_path, output_name="generated", temperature=1.0, top_p=0.9):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    transformer = model.MusicTransformer().to(device)
    if os.path.exists(checkpoint_path):
        transformer.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    transformer.eval()
    
    # Prepare Input
    if emotion not in config.EMOTION_TO_ID:
        print(f"Invalid emotion: {emotion}. Choose from {config.EMOTIONS}")
        return
        
    emotion_id = config.EMOTION_TO_ID[emotion]
    emotion_token = config.TOKEN_OFFSET_EMOTION + emotion_id
    
    # Start with [BOS, EMOTION]
    input_seq = torch.tensor([[config.BOS_TOKEN, emotion_token]], dtype=torch.long).to(device)
    
    generated_tokens = []
    
    print(f"Generating music for emotion: {emotion}...")
    
    with torch.no_grad():
        for _ in range(config.SEQ_LEN):
            output = transformer(input_seq)
            # Get last token logits
            logits = output[:, -1, :] / temperature
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            token_val = next_token.item()
            
            if token_val == config.EOS_TOKEN:
                break
                
            generated_tokens.append(token_val)
            input_seq = torch.cat([input_seq, next_token], dim=1)
            
    # Convert to MIDI
    output_midi_path = os.path.join(config.OUTPUT_DIR, f"{output_name}.mid")
    midi_processor.decode_midi(generated_tokens, output_midi_path)
    print(f"Saved MIDI to {output_midi_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--emotion', type=str, required=True, help='Emotion: Q1 (Joy), Q2 (Tension), Q3 (Sadness), Q4 (Calm)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='generated', help='Output filename')
    args = parser.parse_args()
    
    generate(args.emotion, args.checkpoint, args.output)
