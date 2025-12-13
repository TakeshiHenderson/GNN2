"""
FastAPI backend for Handwritten Equation Solver.
"""
import os
import sys
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Import inference module
from api.inference import EquationRecognizer


# --- Pydantic Models ---
class StrokeData(BaseModel):
    id: str
    points: List[List[float]]


class PredictRequest(BaseModel):
    strokes: List[StrokeData]


class PredictResponse(BaseModel):
    latex: str
    tokens: List[str]
    tex_file: str | None


# --- App Setup ---
app = FastAPI(
    title="Handwritten Equation Solver API",
    description="Recognizes handwritten mathematical equations using a Graph-to-Graph neural network",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
recognizer = None


@app.on_event("startup")
async def load_model():
    """Load the model on startup."""
    global recognizer
    print("Loading equation recognition model...")
    recognizer = EquationRecognizer()
    print("Model ready!")


@app.get("/")
async def root():
    """Serve the frontend."""
    frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'index.html')
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "Handwritten Equation Solver API", "status": "running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": recognizer is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict LaTeX from stroke data.
    
    Expects JSON body:
    ```json
    {
        "strokes": [
            {"id": "0", "points": [[x1, y1], [x2, y2], ...]},
            {"id": "1", "points": [[x1, y1], [x2, y2], ...]}
        ]
    }
    ```
    """
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.strokes:
        return PredictResponse(latex="", tokens=[], tex_file=None)
    
    # Convert to list of dicts for inference
    stroke_data = [{"id": s.id, "points": s.points} for s in request.strokes]
    
    try:
        result = recognizer.predict(stroke_data)
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files for frontend assets
frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
