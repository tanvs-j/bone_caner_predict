import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gradio as gr

from src.model import build_model

# Default to the trained checkpoint we created earlier
CKPT = os.environ.get("BONE_CKPT", r"T:\\bone_can_pre\\models\\efficientnet_b0_folder_best.pt")
MODEL_NAME = "mobilenet_v3_small" if "mobilenet_v3_small" in os.path.basename(CKPT) else "efficientnet_b0"

size = 224
transform = A.Compose([
    A.LongestMaxSize(max_size=size),
    A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(MODEL_NAME, num_classes=2, pretrained=False).to(device)
state = torch.load(CKPT, map_location=device)
model.load_state_dict(state["model"])
model.eval()


def generate_gradcam(img_tensor, original_rgb):
    """Generate Grad-CAM heatmap from the model's last conv layer."""
    # Hook into the last conv layer
    if MODEL_NAME == "efficientnet_b0":
        target_layer = model.features[-1]
    else:
        target_layer = model.features[-1]

    activations = []
    gradients = []

    def fwd_hook(module, inp, out):
        activations.append(out)

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    try:
        model.eval()
        inp = img_tensor.clone().requires_grad_(True)
        logits = model(inp)
        # Back-prop w.r.t. cancer class (index 0)
        model.zero_grad()
        logits[0, 0].backward()

        act = activations[0].detach()
        grad = gradients[0].detach()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    finally:
        fh.remove()
        bh.remove()
        model.eval()

    # Resize heatmap to original image size
    h, w = original_rgb.shape[:2]
    heatmap = cv2.resize(cam, (w, h))

    # Colorize heatmap
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend with original
    overlay = cv2.addWeighted(original_rgb, 0.55, heatmap_color, 0.45, 0)

    # Draw contours around high-activation regions
    mask = (heatmap > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    return Image.fromarray(overlay)


def predict(image: Image.Image):
    if image is None:
        return {"normal": 0.0, "cancer": 0.0}, "⏳ Awaiting image upload…", "—", "—", "—", None
    img = image.convert("RGB")
    img_rgb = np.array(img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    x = transform(image=img_bgr)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0,0].item()
    label = "cancer" if prob >= 0.5 else "normal"
    probs = {"normal": 1.0 - prob, "cancer": prob}

    # Diagnosis detail
    if label == "normal":
        diagnosis = "✅ No Evidence of Malignancy"
        confidence = f"Confidence: {(1.0 - prob):.1%}"
        risk = "🟢 Low Risk"
        lifespan = "Excellent prognosis — No cancer detected"
        highlighted = None  # no highlight for normal
    else:
        diagnosis = "⚠️ Malignancy Detected"
        confidence = f"Confidence: {prob:.1%}"
        if prob < 0.6:
            risk = "🟡 Moderate Risk"
            lifespan = f"Good prognosis (>5 years) — Low-confidence ({prob:.1%})"
        elif prob < 0.8:
            risk = "🟠 Elevated Risk"
            lifespan = f"Moderate prognosis (3-5 years) — Cancer detected ({prob:.1%})"
        else:
            risk = "🔴 High Risk"
            lifespan = f"Guarded prognosis (1-3 years) — High-confidence ({prob:.1%})"
        # Generate Grad-CAM highlighting
        try:
            highlighted = generate_gradcam(x, img_rgb)
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            highlighted = None

    return probs, diagnosis, confidence, risk, lifespan, highlighted


# ── Premium Medical + Developer Theme CSS ───────────────────────────────
custom_css = """
/* ─── Google Font Import ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── Root Variables ─── */
:root {
    --primary: #0ea5e9;
    --primary-dark: #0284c7;
    --accent: #06d6a0;
    --danger: #ef4444;
    --warning: #f59e0b;
    --bg-body: #f0f4f8;
    --bg-card: #ffffff;
    --bg-card-hover: #f8fafc;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --border: #e2e8f0;
    --glow: rgba(14,165,233,0.12);
}

/* ─── Global ─── */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--bg-body) !important;
    max-width: 100% !important;
    min-height: 100vh !important;
    margin: 0 !important;
    padding: 24px 48px !important;
    box-sizing: border-box !important;
}
footer {
    display: none !important;
}

/* ─── Header ─── */
#header-row {
    background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 50%, #38bdf8 100%) !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.25) !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 28px 32px !important;
    margin-bottom: 16px !important;
    position: relative;
    overflow: hidden;
}
#header-row::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--primary), var(--accent), var(--primary));
}

/* ─── Cards ─── */
.card-panel {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    transition: all 0.3s ease !important;
}
.card-panel:hover {
    border-color: var(--primary) !important;
    box-shadow: 0 0 20px var(--glow) !important;
}

/* ─── Upload Area ─── */
.upload-area {
    background: var(--bg-card) !important;
    border: 2px dashed #cbd5e1 !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    min-height: 280px !important;
}
.upload-area:hover {
    border-color: var(--primary) !important;
    box-shadow: 0 0 30px var(--glow) !important;
}

/* ─── Buttons ─── */
.analyze-btn {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 15px rgba(14,165,233,0.3) !important;
}
.analyze-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(14,165,233,0.5) !important;
}

/* ─── Output Fields ─── */
.result-field textarea, .result-field input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important;
    padding: 12px !important;
}

/* ─── Label component ─── */
.label-component {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* ─── Markdown styling ─── */
.markdown-text h1 {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-size: 1.8em !important;
    margin: 0 !important;
    letter-spacing: -0.5px !important;
}
.markdown-text h3 {
    color: rgba(255,255,255,0.85) !important;
    font-weight: 400 !important;
    font-size: 0.95em !important;
    margin: 4px 0 0 0 !important;
}
.markdown-text h4 {
    color: var(--text-primary) !important;
}
.markdown-text p {
    color: var(--text-secondary) !important;
    font-size: 0.9em !important;
    line-height: 1.6 !important;
}
.markdown-text code {
    background: rgba(14,165,233,0.15) !important;
    color: var(--primary) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85em !important;
}

/* ─── Misc ─── */
label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

/* ─── Pulse animation for scan button ─── */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 4px 15px rgba(14,165,233,0.3); }
    50% { box-shadow: 0 4px 25px rgba(14,165,233,0.6); }
}
.analyze-btn {
    animation: pulse-glow 3s ease-in-out infinite !important;
}
.analyze-btn:hover {
    animation: none !important;
}

/* ─── Status badges ─── */
.status-output textarea {
    font-size: 15px !important;
    font-weight: 600 !important;
    text-align: center !important;
    letter-spacing: 0.3px !important;
}
"""


# ── Build the Gradio app ────────────────────────────────────────────────
with gr.Blocks(title="BoneScan AI — Cancer Classifier") as demo:

    # Header
    with gr.Row(elem_id="header-row"):
        gr.Markdown("""
# 🦴 BoneScan AI
### KNN and K-Means Bone Cancer Classification System using ML
        """, elem_classes=["markdown-text"])

    with gr.Row():
        # ─── Left Column: Upload ────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("#### 📤 Upload X-Ray", elem_classes=["markdown-text"])
            inp = gr.Image(type="pil", label="Bone X-Ray Image",
                           elem_classes=["upload-area"],
                           height=300)
            btn = gr.Button("🔬 Analyze Scan", elem_classes=["analyze-btn"],
                            variant="primary", size="lg")

        # ─── Middle Column: Results ─────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("#### 📊 Analysis Results", elem_classes=["markdown-text"])

            out_diag = gr.Textbox(label="🩺 Diagnosis", interactive=False,
                                  elem_classes=["result-field", "status-output"])
            out_probs = gr.Label(num_top_classes=2, label="📈 Class Probabilities",
                                 elem_classes=["label-component"])

            with gr.Row():
                out_conf = gr.Textbox(label="🎯 Confidence", interactive=False,
                                      elem_classes=["result-field"])
                out_risk = gr.Textbox(label="⚡ Risk Level", interactive=False,
                                      elem_classes=["result-field"])

            out_life = gr.Textbox(label="📅 Survival Estimate", interactive=False,
                                  elem_classes=["result-field"])

        # ─── Right Column: Cancer Highlight ─────────────────────
        with gr.Column(scale=1):
            gr.Markdown("#### 🔍 Cancer Region Highlight", elem_classes=["markdown-text"])
            out_highlight = gr.Image(label="Grad-CAM Heatmap",
                                     type="pil",
                                     height=380)
            gr.Markdown("*Red/warm areas indicate regions the model focuses on for cancer detection.*",
                        elem_classes=["markdown-text"])

    btn.click(predict, inputs=inp, outputs=[out_probs, out_diag, out_conf, out_risk, out_life, out_highlight])



if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=custom_css, theme=gr.themes.Base())
