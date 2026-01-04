import os

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'midis')
LABEL_FILE = os.path.join(ROOT_DIR, 'EMOPIA_1.0', 'label.csv')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
SOUNDFONT_PATH = os.path.join(ROOT_DIR, 'soundfont.sf2')

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tokenization
# We will use a simple event-based representation
# Events: Note-On (Pitch, Velocity), Time-Shift, Note-Off (or Duration)
# For simplicity, we'll use: Pitch, Duration, TimeShift (Velocity fixed or simplified)

# Ranges
MIN_PITCH = 21
MAX_PITCH = 108
NUM_PITCHES = MAX_PITCH - MIN_PITCH + 1

# Quantization
TICKS_PER_BEAT = 480
BEATS_PER_BAR = 4 # Assuming 4/4 mostly
BIN_SIZE = TICKS_PER_BEAT // 4 # 16th note quantization

# Vocabulary definitions
# 0-3: Special
# 4-(4+NUM_PITCHES): Note On
# ... Time Shift
# ... Duration

PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3

# Offsets for different event types
TOKEN_OFFSET_PITCH = 4
TOKEN_OFFSET_TIME = TOKEN_OFFSET_PITCH + NUM_PITCHES
NUM_TIME_SHIFTS = 100 # 100 steps of time shift
TOKEN_OFFSET_DURATION = TOKEN_OFFSET_TIME + NUM_TIME_SHIFTS
NUM_DURATIONS = 64 # 64 duration bins

TOKEN_OFFSET_EMOTION = TOKEN_OFFSET_DURATION + NUM_DURATIONS
NUM_EMOTIONS = 4

VOCAB_SIZE = TOKEN_OFFSET_EMOTION + NUM_EMOTIONS

# Emotion Labels
EMOTIONS = ['Q1', 'Q2', 'Q3', 'Q4']
EMOTION_TO_ID = {e: i for i, e in enumerate(EMOTIONS)}

# Model Hyperparameters
EMBED_DIM = 256
N_LAYERS = 4
N_HEADS = 4
FF_DIM = 1024
DROPOUT = 0.1
SEQ_LEN = 512 # Max sequence length

# Training Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 20
GRAD_CLIP = 1.0
