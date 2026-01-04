from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys
import os
import uuid

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src import generate, config

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    emotion: str # Joy, Tension, Sadness, Calm

@app.post("/generate")
async def generate_music(request: GenerateRequest):
    emotion_map = {
        "Joy": "Q1",
        "Tension": "Q2",
        "Sadness": "Q3",
        "Calm": "Q4"
    }
    
    emotion_code = emotion_map.get(request.emotion)
    if not emotion_code:
        raise HTTPException(status_code=400, detail="Invalid emotion")

    # Find the latest checkpoint
    if not os.path.exists(config.CHECKPOINT_DIR):
         raise HTTPException(status_code=500, detail="Checkpoint directory not found")

    checkpoints = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.endswith('.pt')]
    if not checkpoints:
        raise HTTPException(status_code=500, detail="No checkpoints found")
    
    # Sort by epoch number (assuming format model_epoch_X.pt)
    try:
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    except:
        pass 
        
    latest_checkpoint = os.path.join(config.CHECKPOINT_DIR, checkpoints[-1])
    
    output_id = str(uuid.uuid4())
    
    try:
        # Call the generate function
        generate.generate(
            emotion=emotion_code,
            checkpoint_path=latest_checkpoint,
            output_name=output_id
        )
        
        midi_path = os.path.join(config.OUTPUT_DIR, f"{output_id}.mid")
        
        if os.path.exists(midi_path):
            return {"url": f"http://localhost:8000/download/{output_id}.mid", "type": "midi"}
        else:
            raise HTTPException(status_code=500, detail="Generation failed - No output file created")
            
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(config.OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
