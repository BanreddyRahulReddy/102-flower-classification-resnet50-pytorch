# 🌸 102 Flower Classification with PyTorch & ResNet50

A deep learning image classifier that identifies **102 flower species** using transfer learning on a pretrained ResNet50 model. Trained on the Oxford 102 Category Flower Dataset with a two-phase training strategy, OneCycleLR scheduling, and comprehensive evaluation tools.

---

## 📌 Overview

This project fine-tunes a **ResNet50** backbone (pretrained on ImageNet) for fine-grained flower classification across 102 categories. Training is split into two phases:

- **Phase 1** — Only the custom classifier head is trained while the backbone is frozen (fast convergence)
- **Phase 2** — Full fine-tuning with all layers unfrozen and a much lower learning rate (squeezes out extra accuracy)

The notebook also includes rich post-training evaluation: Top-5 accuracy, a 102×102 confusion matrix heatmap, most-confused flower pairs, and per-class accuracy breakdown.

---

## 📂 Dataset

**Oxford 102 Category Flower Dataset**
🔗 https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

- 102 flower categories
- ~8,000 images split into `train`, `valid`, and `test` sets
- `cat_to_name.json` maps category indices to human-readable flower names

---

## 🏗️ Model Architecture

| Component | Details |
|---|---|
| Backbone | ResNet50 (pretrained on ImageNet) |
| Classifier Head | Custom FC layers replacing the default final layer |
| Frozen Layers | All except last 20 params during Phase 1 |
| Dropout | 0.3 (in classifier head) |
| Output | 102 classes (softmax) |

---

## ⚙️ Training Configuration

| Hyperparameter | Phase 1 | Phase 2 |
|---|---|---|
| Epochs | 15 | 5 |
| Max Learning Rate | 3e-3 | 1e-4 |
| Optimizer | AdamW | AdamW |
| Scheduler | OneCycleLR | OneCycleLR |
| Weight Decay | 1e-2 | 1e-2 |
| Label Smoothing | 0.1 | 0.1 |
| Gradient Clipping | 1.0 | 1.0 |
| Batch Size | 128 | 128 |
| Image Size | 256×256 | 256×256 |

---

## 🔄 Data Augmentation

**Training:**
- Padding with reflect mode (8px)
- Random rotation (±30°)
- Random resized crop (256×256)
- Random horizontal flip
- Color jitter (brightness, contrast, saturation ±0.2)
- Normalized with ImageNet stats

**Validation / Test:**
- Resize and center crop only
- Normalized with ImageNet stats (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)

---

## 📊 Evaluation

The notebook includes the following evaluation tools after training:

- ✅ **Top-1 & Top-5 Accuracy** — standard classification metrics
- 🟥 **Confusion Matrix** — full 102×102 heatmap visualization
- 🔁 **Most Confused Pairs** — top 10 flower pairs the model mixes up most
- 📉 **Per-Class Accuracy** — color-coded bar chart for all 102 classes

---

## 📁 File Structure

```
├── flower_classification_PyTorch.ipynb   # Main notebook
├── cat_to_name.json                      # Category index → flower name mapping
├── best_model.pth                        # Best checkpoint saved during training
└── flower-classifier-final.pth          # Final model with metadata (for inference)
```

---

## 🚀 How to Run

1. **Open the notebook** in Google Colab

2. **Mount Google Drive** and ensure your dataset is at:
   ```
   /content/drive/MyDrive/flower_data/
   ├── train/   (subfolders 1–102)
   ├── valid/   (subfolders 1–102)
   └── test/    (single subfolder 0/)
   ```

3. **Copy dataset to local Colab SSD** (cell provided in notebook — speeds up data loading significantly)

4. **Run training cells** in order:
   - Phase 1: Frozen backbone training (15 epochs)
   - Phase 2: Full fine-tuning (5 epochs)

5. **Load best model and evaluate:**
   ```python
   model.load_state_dict(torch.load('best_model.pth', map_location=device))
   ```

6. **Run evaluation cells** for confusion matrix, per-class accuracy, etc.

---

## 📦 Requirements

```
torch
torchvision
numpy
matplotlib
seaborn
scikit-learn
pandas
tqdm
```

Install with:
```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn pandas tqdm
```

---

## 🔮 Inference

```python
def predict_image(img, model, dataset):
    xb = img.unsqueeze(0).to(device)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return dataset.classes[preds[0].item()]

img, label = testing_imagefolder[0]
print('Predicted:', predict_image(img, model, training_imagefolder))
```

---

## 💾 Saving & Loading

**Save:**
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'num_classes': len(training_imagefolder.classes),
    'class_to_idx': training_imagefolder.class_to_idx,
    'img_size': 256,
    'architecture': 'resnet50',
}
torch.save(checkpoint, 'flower-classifier-final.pth')
```

**Load:**
```python
checkpoint = torch.load('flower-classifier-final.pth', map_location=device)
model = FlowersModel(checkpoint['num_classes']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## 📚 References

- [Oxford 102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
- [PyTorch ResNet50 Documentation](https://pytorch.org/vision/stable/models/resnet.html)
- [cat_to_name.json source](https://github.com/nirajpandkar/flowers-classification-pytorch)

---
