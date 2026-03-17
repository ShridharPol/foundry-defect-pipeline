"""
Foundry Defect Detection — Model Serving API
Accepts a casting image, returns defective / ok prediction.
Developed at Hamdan InfoCom, Belagavi.
"""

import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Dict

# --- Config ---
MODEL_PATH = "best_mobilenetv2.pth"
CLASSES    = ["def_front", "ok_front"]
DEVICE     = torch.device("cpu")

# --- Response Models ---
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prediction": "def_front",
                    "confidence": 0.9998,
                    "probabilities": {
                        "def_front": 0.9998,
                        "ok_front": 0.0002
                    }
                }
            ]
        }
    }

class HealthResponse(BaseModel):
    status: str

class RootResponse(BaseModel):
    status: str
    model: str
    classes: list
    description: str
    developed_by: str
    endpoints: Dict[str, str]

# --- Load model ---
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model

model = load_model()

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# --- App ---
description = """
## Foundry Defect Detection API

Built by **Hamdan InfoCom, Belagavi** as a proof-of-concept quality inspection system
for iron casting manufacturers in the Belagavi industrial cluster.

### Model Performance
| Metric | Value |
|---|---|
| Accuracy | 100% |
| Precision | 1.00 |
| Recall | 1.00 |
| F1-Score | 1.00 |
| Val set size | 715 images |

### Classes
| Class | Meaning |
|---|---|
| `def_front` | Defective casting — should be rejected |
| `ok_front` | OK casting — passes quality inspection |
"""

app = FastAPI(
    title="Foundry Defect Detection API",
    description=description,
    version="1.0.0",
    contact={
        "name": "Hamdan InfoCom",
        "url": "https://github.com/ShridharPol/foundry-defect-pipeline",
    },
)

# --- Demo UI ---
@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def demo_ui():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Foundry Defect Detection — Hamdan InfoCom</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Segoe UI', Arial, sans-serif;
      background: #f0f2f5;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: linear-gradient(135deg, #1a1a2e, #0f3460);
      color: white;
      padding: 24px 40px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    header h1 { font-size: 20px; font-weight: 700; }
    header p  { font-size: 12px; color: #a8b2d8; margin-top: 4px; }
    .badge {
      background: #e94560;
      color: white;
      font-size: 11px;
      font-weight: 700;
      padding: 4px 12px;
      border-radius: 20px;
      letter-spacing: 1px;
      text-transform: uppercase;
    }
    main {
      flex: 1;
      display: flex;
      gap: 24px;
      padding: 40px;
      max-width: 1000px;
      margin: 0 auto;
      width: 100%;
    }
    .card {
      background: white;
      border-radius: 12px;
      padding: 28px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    .upload-card { flex: 1; }
    .result-card { flex: 1; }
    h2 {
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 2px;
      color: #0f3460;
      margin-bottom: 20px;
      font-weight: 700;
    }
    .drop-zone {
      border: 2px dashed #d0d7e8;
      border-radius: 10px;
      padding: 40px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s;
      background: #f8f9ff;
    }
    .drop-zone:hover, .drop-zone.dragover {
      border-color: #0f3460;
      background: #eef1fb;
    }
    .drop-zone .icon { font-size: 40px; margin-bottom: 12px; }
    .drop-zone p { font-size: 14px; color: #666; }
    .drop-zone span { color: #0f3460; font-weight: 600; }
    #file-input { display: none; }
    #preview-wrap {
      display: none;
      margin-top: 16px;
      text-align: center;
    }
    #preview {
      max-width: 100%;
      max-height: 220px;
      border-radius: 8px;
      border: 1px solid #e0e4f0;
    }
    #filename {
      font-size: 12px;
      color: #999;
      margin-top: 8px;
    }
    .btn {
      display: block;
      width: 100%;
      margin-top: 20px;
      padding: 14px;
      background: #0f3460;
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }
    .btn:hover { background: #1a1a2e; }
    .btn:disabled { background: #b0b8cc; cursor: not-allowed; }

    /* Result */
    .placeholder {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 200px;
      color: #b0b8cc;
      font-size: 14px;
      gap: 12px;
    }
    .placeholder .icon { font-size: 40px; }
    .result-content { display: none; }
    .verdict {
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 20px;
      border-radius: 10px;
      margin-bottom: 20px;
    }
    .verdict.defective { background: #fff0f3; border: 1px solid #ffccd5; }
    .verdict.ok        { background: #f0fff4; border: 1px solid #b7ebc8; }
    .verdict-icon { font-size: 36px; }
    .verdict-label { font-size: 22px; font-weight: 700; }
    .verdict.defective .verdict-label { color: #e94560; }
    .verdict.ok        .verdict-label { color: #22a55b; }
    .verdict-sub { font-size: 13px; color: #888; margin-top: 2px; }

    .prob-row { margin-bottom: 14px; }
    .prob-label {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      color: #444;
      margin-bottom: 6px;
      font-weight: 600;
    }
    .bar-bg {
      background: #f0f2f5;
      border-radius: 6px;
      height: 10px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: 6px;
      transition: width 0.6s ease;
    }
    .bar-def { background: #e94560; }
    .bar-ok  { background: #22a55b; }

    .meta-row {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: #888;
      margin-top: 20px;
      padding-top: 16px;
      border-top: 1px solid #eee;
    }

    .spinner {
      display: none;
      text-align: center;
      padding: 60px 0;
      color: #888;
      font-size: 14px;
    }
    .spinner::before {
      content: '';
      display: block;
      width: 36px;
      height: 36px;
      border: 3px solid #e0e4f0;
      border-top-color: #0f3460;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin: 0 auto 16px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    footer {
      text-align: center;
      padding: 16px;
      font-size: 12px;
      color: #aaa;
    }
  </style>
</head>
<body>

<header>
  <div>
    <h1>Foundry Defect Detection</h1>
    <p>MobileNetV2 · Iron Casting Quality Inspection · Belagavi Industrial Cluster</p>
  </div>
  <div class="badge">Hamdan InfoCom</div>
</header>

<main>
  <!-- Upload -->
  <div class="card upload-card">
    <h2>Upload Casting Image</h2>
    <div class="drop-zone" id="drop-zone">
      <div class="icon">🏭</div>
      <p><span>Click to upload</span> or drag & drop</p>
      <p style="margin-top:6px;font-size:12px;">JPG or PNG · Any resolution</p>
    </div>
    <input type="file" id="file-input" accept="image/*"/>
    <div id="preview-wrap">
      <img id="preview" src="" alt="Preview"/>
      <div id="filename"></div>
    </div>
    <button class="btn" id="predict-btn" disabled>Classify Image</button>
  </div>

  <!-- Result -->
  <div class="card result-card">
    <h2>Prediction Result</h2>
    <div class="placeholder" id="placeholder">
      <div class="icon">🔍</div>
      <span>Upload an image to see the prediction</span>
    </div>
    <div class="spinner" id="spinner">Classifying image...</div>
    <div class="result-content" id="result-content">
      <div class="verdict" id="verdict">
        <div class="verdict-icon" id="verdict-icon"></div>
        <div>
          <div class="verdict-label" id="verdict-label"></div>
          <div class="verdict-sub" id="verdict-sub"></div>
        </div>
      </div>
      <div class="prob-row">
        <div class="prob-label">
          <span>Defective (def_front)</span>
          <span id="def-pct"></span>
        </div>
        <div class="bar-bg"><div class="bar-fill bar-def" id="def-bar" style="width:0%"></div></div>
      </div>
      <div class="prob-row">
        <div class="prob-label">
          <span>OK (ok_front)</span>
          <span id="ok-pct"></span>
        </div>
        <div class="bar-bg"><div class="bar-fill bar-ok" id="ok-bar" style="width:0%"></div></div>
      </div>
      <div class="meta-row">
        <span>Model: MobileNetV2</span>
        <span>Trained on 6,633 foundry images</span>
        <span>Accuracy: 100%</span>
      </div>
    </div>
  </div>
</main>

<footer>Foundry Defect Detection API · Hamdan InfoCom · Belagavi, India · <a href="/docs" style="color:#0f3460">API Docs</a></footer>

<script>
  const dropZone   = document.getElementById('drop-zone');
  const fileInput  = document.getElementById('file-input');
  const preview    = document.getElementById('preview');
  const previewWrap= document.getElementById('preview-wrap');
  const filename   = document.getElementById('filename');
  const predictBtn = document.getElementById('predict-btn');
  let selectedFile = null;

  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));

  function handleFile(file) {
    if (!file) return;
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = e => {
      preview.src = e.target.result;
      previewWrap.style.display = 'block';
      filename.textContent = file.name;
      predictBtn.disabled = false;
    };
    reader.readAsDataURL(file);
    resetResult();
  }

  predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    predictBtn.disabled = true;
    showSpinner();

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res  = await fetch('/predict', { method: 'POST', body: formData });
      const data = await res.json();
      showResult(data);
    } catch(e) {
      alert('Error calling API. Is the server running?');
    } finally {
      predictBtn.disabled = false;
    }
  });

  function showSpinner() {
    document.getElementById('placeholder').style.display    = 'none';
    document.getElementById('result-content').style.display = 'none';
    document.getElementById('spinner').style.display        = 'block';
  }

  function resetResult() {
    document.getElementById('placeholder').style.display    = 'flex';
    document.getElementById('result-content').style.display = 'none';
    document.getElementById('spinner').style.display        = 'none';
  }

  function showResult(data) {
    document.getElementById('spinner').style.display        = 'none';
    document.getElementById('result-content').style.display = 'block';

    const isDefective = data.prediction === 'def_front';
    const verdict     = document.getElementById('verdict');
    verdict.className = 'verdict ' + (isDefective ? 'defective' : 'ok');
    document.getElementById('verdict-icon').textContent  = isDefective ? '❌' : '✅';
    document.getElementById('verdict-label').textContent = isDefective ? 'DEFECTIVE' : 'OK';
    document.getElementById('verdict-sub').textContent   = isDefective
      ? 'This casting should be rejected'
      : 'This casting passes quality inspection';

    const defPct = (data.probabilities.def_front * 100).toFixed(2) + '%';
    const okPct  = (data.probabilities.ok_front  * 100).toFixed(2) + '%';
    document.getElementById('def-pct').textContent        = defPct;
    document.getElementById('ok-pct').textContent         = okPct;
    document.getElementById('def-bar').style.width        = defPct;
    document.getElementById('ok-bar').style.width         = okPct;
  }
</script>
</body>
</html>
""")

# --- API Endpoints ---
@app.get("/", response_model=RootResponse, summary="API Overview")
def root():
    return {
        "status": "ok",
        "model": "MobileNetV2",
        "classes": CLASSES,
        "description": "Casting defect classifier — def_front vs ok_front",
        "developed_by": "Hamdan InfoCom, Belagavi",
        "endpoints": {
            "GET  /":        "API overview",
            "GET  /health":  "Health check",
            "GET  /ui":      "Demo UI",
            "POST /predict": "Classify a casting image",
        }
    }

@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse, summary="Classify a Casting Image")
async def predict(file: UploadFile = File(..., description="Casting image — JPG or PNG")):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Please upload a JPG or PNG image."
        )
    contents = await file.read()
    img      = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor   = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs  = model(tensor)
        probs    = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()

    return JSONResponse({
        "prediction":  CLASSES[pred_idx],
        "confidence":  round(probs[pred_idx].item(), 4),
        "probabilities": {
            CLASSES[0]: round(probs[0].item(), 4),
            CLASSES[1]: round(probs[1].item(), 4),
        }
    })