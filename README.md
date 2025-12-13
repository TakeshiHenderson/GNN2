# Handwritten Equation Solver

A prototype system for recognizing handwritten mathematical equations using a Graph-to-Graph neural network. Draw equations on a browser-based whiteboard and get LaTeX output.

## Quick Start

```bash
# Install dependencies (if not already done)
uv add fastapi uvicorn python-multipart

# Start the server
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

## Usage

1. **Draw** a mathematical equation on the canvas using mouse or touch
2. **Click "Recognize"** to send strokes to the model
3. **View results**: LaTeX output, raw tokens, and saved `.tex` file path

## API

### POST /predict

Recognize handwritten strokes.

**Request:**
```json
{
  "strokes": [
    {"id": "0", "points": [[x1, y1], [x2, y2], ...]},
    {"id": "1", "points": [[x1, y1], [x2, y2], ...]}
  ]
}
```

**Response:**
```json
{
  "latex": "x + 1",
  "tokens": ["x(Right)", "+(Right)", "1(Right)", "<EOS>(-)"],
  "tex_file": "/path/to/output/equation_20231213_120000.tex"
}
```

### GET /health

Health check endpoint.

## Project Structure

```
GNN2/
├── api/
│   ├── main.py          # FastAPI server
│   ├── inference.py     # Model loading & prediction
│   └── requirements.txt # API dependencies
├── frontend/
│   ├── index.html       # Whiteboard UI
│   ├── style.css        # Styling
│   └── app.js           # Stroke capture & API calls
├── output/              # Generated .tex files
├── checkpoints_6/       # Trained model weights
└── src/
    ├── config.py        # Model configuration
    ├── gnn_model/       # GNN encoder/decoder
    ├── source_graph/    # Stroke graph construction
    └── vocab.json       # Symbol & relation vocabularies
```

## Requirements

- Python 3.10
- PyTorch
- FastAPI, Uvicorn
- Dependencies managed by `uv` (see `pyproject.toml`)

## Model

The system uses a Graph-to-Graph neural network trained on the CROHME dataset. The model:
1. Converts strokes to a graph with spatial (LOS) and temporal edges
2. Encodes the graph using a GNN encoder
3. Decodes symbols and spatial relations autoregressively
4. Converts output tokens to LaTeX format

## Output

Recognized equations are:
- Returned as JSON (LaTeX string + tokens)
- Saved as `.tex` files in the `output/` directory
