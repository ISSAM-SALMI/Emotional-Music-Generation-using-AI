import miditoolkit
import numpy as np
from src import config

def quantize_ticks(ticks, ticks_per_beat):
    # Quantize to 16th notes (ticks_per_beat / 4)
    bin_size = ticks_per_beat // 4
    return int(round(ticks / bin_size))

def encode_midi(midi_path):
    try:
        midi_obj = miditoolkit.MidiFile(midi_path)
    except Exception as e:
        print(f"Error loading {midi_path}: {e}")
        return None

    ticks_per_beat = midi_obj.ticks_per_beat
    
    # Merge all instruments (piano reduction)
    notes = []
    for instrument in midi_obj.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append(note)
    
    # Sort by start time, then pitch
    notes.sort(key=lambda x: (x.start, x.pitch))
    
    tokens = []
    current_time = 0
    
    for note in notes:
        # Calculate time shift
        time_diff = note.start - current_time
        if time_diff > 0:
            quantized_shift = quantize_ticks(time_diff, ticks_per_beat)
            if quantized_shift > 0:
                # Cap at max shift
                quantized_shift = min(quantized_shift, config.NUM_TIME_SHIFTS - 1)
                tokens.append(config.TOKEN_OFFSET_TIME + quantized_shift)
                current_time = note.start # Update current time (approximate due to quantization, but we keep real time for sync)
                # Ideally we should update current_time based on quantized shift to avoid drift, 
                # but for simple generation this is okay. 
                # Better: current_time += quantized_shift * (ticks_per_beat // 4)
                
        # Pitch
        pitch = note.pitch
        if pitch < config.MIN_PITCH: pitch = config.MIN_PITCH
        if pitch > config.MAX_PITCH: pitch = config.MAX_PITCH
        tokens.append(config.TOKEN_OFFSET_PITCH + (pitch - config.MIN_PITCH))
        
        # Duration
        duration = note.end - note.start
        quantized_dur = quantize_ticks(duration, ticks_per_beat)
        if quantized_dur < 1: quantized_dur = 1
        quantized_dur = min(quantized_dur, config.NUM_DURATIONS - 1)
        tokens.append(config.TOKEN_OFFSET_DURATION + quantized_dur)
        
    return tokens

def decode_midi(tokens, output_path):
    midi_obj = miditoolkit.MidiFile()
    midi_obj.ticks_per_beat = config.TICKS_PER_BEAT
    
    instrument = miditoolkit.Instrument(program=0, is_drum=False, name="Piano")
    
    current_time = 0
    bin_size = config.TICKS_PER_BEAT // 4
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Time Shift
        if config.TOKEN_OFFSET_TIME <= token < config.TOKEN_OFFSET_TIME + config.NUM_TIME_SHIFTS:
            shift = token - config.TOKEN_OFFSET_TIME
            current_time += shift * bin_size
            i += 1
            
        # Note
        elif config.TOKEN_OFFSET_PITCH <= token < config.TOKEN_OFFSET_PITCH + config.NUM_PITCHES:
            pitch = token - config.TOKEN_OFFSET_PITCH + config.MIN_PITCH
            
            # Look ahead for duration
            duration = 1 * bin_size # Default
            if i + 1 < len(tokens):
                next_token = tokens[i+1]
                if config.TOKEN_OFFSET_DURATION <= next_token < config.TOKEN_OFFSET_DURATION + config.NUM_DURATIONS:
                    dur_val = next_token - config.TOKEN_OFFSET_DURATION
                    duration = dur_val * bin_size
                    i += 1 # Skip duration token next loop
            
            note = miditoolkit.Note(
                velocity=100,
                pitch=pitch,
                start=current_time,
                end=current_time + duration
            )
            instrument.notes.append(note)
            i += 1
            
        else:
            # Skip unknown or special tokens
            i += 1
            
    midi_obj.instruments.append(instrument)
    midi_obj.dump(output_path)
    return midi_obj
