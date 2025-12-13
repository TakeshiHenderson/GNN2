/**
 * Handwritten Equation Solver - Frontend JavaScript
 * Handles canvas drawing and API communication
 */

// DOM Elements
const canvas = document.getElementById('whiteboard');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const submitBtn = document.getElementById('submitBtn');
const resultSection = document.getElementById('resultSection');
const latexOutput = document.getElementById('latexOutput');
const tokensOutput = document.getElementById('tokensOutput');
const fileOutput = document.getElementById('fileOutput');
const statusEl = document.getElementById('status');

// State
let isDrawing = false;
let currentStroke = [];
let strokes = [];
let strokeId = 0;

// Canvas setup
function setupCanvas() {
    // Set canvas size based on container
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    
    // Drawing style
    ctx.strokeStyle = '#2c3e50';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // Fill white background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Get point from event (mouse or touch)
function getPoint(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    if (e.touches && e.touches.length > 0) {
        return {
            x: (e.touches[0].clientX - rect.left) * scaleX,
            y: (e.touches[0].clientY - rect.top) * scaleY
        };
    }
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

// Drawing handlers
function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    currentStroke = [];
    const point = getPoint(e);
    currentStroke.push([point.x, point.y]);
    ctx.beginPath();
    ctx.moveTo(point.x, point.y);
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const point = getPoint(e);
    currentStroke.push([point.x, point.y]);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
}

function stopDrawing(e) {
    if (!isDrawing) return;
    e.preventDefault();
    isDrawing = false;
    
    if (currentStroke.length > 1) {
        strokes.push({
            id: String(strokeId++),
            points: currentStroke
        });
    }
    currentStroke = [];
}

// Clear canvas
function clearCanvas() {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    strokes = [];
    strokeId = 0;
    resultSection.style.display = 'none';
    setStatus('');
}

// Update status
function setStatus(message, type = '') {
    statusEl.textContent = message;
    statusEl.className = 'status ' + type;
}

// Submit strokes for recognition
async function submitStrokes() {
    if (strokes.length === 0) {
        setStatus('Please draw something first', 'error');
        return;
    }
    
    submitBtn.disabled = true;
    setStatus('Recognizing equation...', 'loading');
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ strokes: strokes })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Recognition failed');
        }
        
        const result = await response.json();
        displayResult(result);
        setStatus('Recognition complete!', 'success');
        
    } catch (error) {
        console.error('Error:', error);
        setStatus('Error: ' + error.message, 'error');
    } finally {
        submitBtn.disabled = false;
    }
}

// Display recognition result
function displayResult(result) {
    resultSection.style.display = 'block';
    
    latexOutput.textContent = result.latex || '(no output)';
    tokensOutput.textContent = result.tokens ? result.tokens.join(', ') : '(no tokens)';
    fileOutput.textContent = result.tex_file || '(not saved)';
}

// Event listeners
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseleave', stopDrawing);

canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);
canvas.addEventListener('touchcancel', stopDrawing);

clearBtn.addEventListener('click', clearCanvas);
submitBtn.addEventListener('click', submitStrokes);

// Initialize
window.addEventListener('load', setupCanvas);
window.addEventListener('resize', () => {
    // Redraw after resize (strokes will be lost)
    setupCanvas();
    strokes = [];
    strokeId = 0;
});
