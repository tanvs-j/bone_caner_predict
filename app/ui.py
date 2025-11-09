import os
import numpy as np
import torch
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gradio as gr

from src.model import build_model

# Default to the trained checkpoint we created earlier
CKPT = os.environ.get("BONE_CKPT", r"T:\\bone_can_pre\\models\\mobilenet_v3_small_best.pt")
MODEL_NAME = "mobilenet_v3_small" if "mobilenet_v3_small" in os.path.basename(CKPT) else "efficientnet_b0"

size = 256
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


def predict(image: Image.Image):
    if image is None:
        return {"normal": 0.0, "cancer": 0.0}, "Please upload an image.", "N/A"
    img = image.convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    x = transform(image=img_bgr)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0,1].item()
    label = "cancer" if prob >= 0.5 else "normal"
    probs = {"normal": 1.0 - prob, "cancer": prob}
    # Lifespan note: requires survival model + labels; not available in current model
    lifespan = "Unavailable (survival model not trained). Provide survival labels to enable this."
    return probs, f"Prediction: {label}", lifespan


with gr.Blocks(title="Bone Cancer Classifier") as demo:
    gr.Markdown("# Bone Cancer Classifier\nUpload an X-ray/CT-type image. The model predicts cancer vs. normal.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Upload image")
            btn = gr.Button("Predict")
        with gr.Column():
            out_probs = gr.Label(num_top_classes=2, label="Class probabilities")
            out_pred = gr.Textbox(label="Prediction", interactive=False)
            out_life = gr.Textbox(label="Estimated lifespan", interactive=False)
    btn.click(predict, inputs=inp, outputs=[out_probs, out_pred, out_life])
    gr.Markdown("""
### Notes
- This demo uses a CNN classifier trained on your dataset for binary prediction only.
- Lifespan estimation is not possible without survival labels (time-to-event and censoring). If you provide them, I will add a survival head (e.g., DeepSurv) and enable this field.
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
