import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# -------------------------------
# 1. Data Preprocessing
# -------------------------------
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'bone_cancer_dataset'
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
    'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val']),
    'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False),
    'test': DataLoader(image_datasets['test'], batch_size=1, shuffle=False)
}

class_names = image_datasets['train'].classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 2. Model: Transfer Learning (ResNet50)
# -------------------------------
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: normal, cancer
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Only train classifier first

# -------------------------------
# 3. Training Function
# -------------------------------
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

# Train the model
print("Training the model...")
model = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)

# Save model
torch.save(model.state_dict(), 'bone_cancer_model.pth')
print("Model saved as 'bone_cancer_model.pth'")

# -------------------------------
# 4. Grad-CAM for Cancer Localization
# -------------------------------
def get_gradcam(model, image_path, class_idx=None, device='cuda'):
    model.eval()
    
    # Hook to get gradients and activations
    grad_cam = None
    def save_grad(module, grad_in, grad_out):
        grad_cam = grad_out[0].cpu().data.numpy().squeeze()
        return grad_cam

    def save_activation(module, input, output):
        activation = output.cpu().data.numpy().squeeze()
        return activation

    # Register hooks
    final_layer = model._modules.get('layer4')
    grad_cam = None
    activation = None

    def save_grad(module, grad_in, grad_out):
        nonlocal grad_cam
        grad_cam = grad_out[0].cpu().data.numpy()

    def save_activation(module, input, output):
        nonlocal activation
        activation = output.cpu().data.numpy()

    final_layer.register_forward_hook(save_activation)
    final_layer.register_backward_hook(save_grad)

    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    input_tensor = data_transforms['test'](img).unsqueeze(0).to(device)

    # Forward
    output = model(input_tensor)
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()

    # Backward
    model.zero_grad()
    class_score = output[:, class_idx]
    class_score.backward()

    # Generate heatmap
    grads = grad_cam[0]  # (2048, 7, 7)
    activations = activation[0]  # (2048, 7, 7)
    weights = np.mean(grads, axis=(1, 2))  # Global average pooling
    cam = np.zeros(activations.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activations[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    return cam, class_idx

# -------------------------------
# 5. Predict & Visualize on Test Image
# -------------------------------
def predict_and_visualize(image_path, model, class_names):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    input_tensor = data_transforms['test'](img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0][pred_class].item()

    # Get Grad-CAM
    heatmap, _ = get_gradcam(model, image_path, class_idx=pred_class, device=device)

    # Overlay heatmap
    img_np = np.array(img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    superimposed = heatmap * 0.4 + img_np / 255
    superimposed = superimposed / superimposed.max()

    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_np)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Prediction: {class_names[pred_class]} ({confidence:.2f})")
    plt.imshow(superimposed)
    plt.axis('off')
    plt.show()

# -------------------------------
# 6. Test on a single image
# -------------------------------
# Load trained model
model.load_state_dict(torch.load('bone_cancer_model.pth'))
model.eval()

# Test on a new image
test_image_path = 'bone_cancer_dataset/test/cancer/sample_cancer.jpg'  # Change path
predict_and_visualize(test_image_path, model, class_names)