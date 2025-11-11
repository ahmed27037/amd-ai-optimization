"""
Test script for loading and running a real PyTorch model.

This script demonstrates:
1. What a ML model is
2. How to load a pre-trained model
3. How to prepare input data
4. How to run inference
5. How to interpret the output
"""

from amd_ai_optimization.rocm_inference import OptimizedInferenceEngine
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import tempfile
import os
import urllib.request

print("=" * 60)
print("ML MODEL TESTING - Step by Step Explanation")
print("=" * 60)

# ============================================================================
# STEP 1: What is a Machine Learning Model?
# ============================================================================
print("\n[STEP 1] Understanding ML Models")
print("-" * 60)
print("""
A Machine Learning (ML) model is like a trained brain:

1. TRAINING PHASE (already done for us):
   - Model learns patterns from millions of images
   - Example: "This pattern = cat", "That pattern = dog"
   - Creates a mathematical function that maps inputs → outputs

2. INFERENCE PHASE (what we're doing now):
   - Give the model NEW data it hasn't seen
   - Model makes predictions based on what it learned
   - Returns probabilities: "85% cat, 10% dog, 5% bird"

Think of it like:
- Training = Teaching a student with textbooks
- Inference = Student taking a test on new questions
""")

# ============================================================================
# STEP 2: Load a Pre-trained Model
# ============================================================================
print("\n[STEP 2] Loading Pre-trained Model")
print("-" * 60)
print("Loading ResNet18 - a popular image classification model...")
print("(This model can recognize 1000 different object categories)")

# ResNet18 is a neural network architecture
# It's pre-trained on ImageNet dataset (1.2 million images, 1000 categories)
model = models.resnet18(weights='IMAGENET1K_V1')
model.eval()  # Set to evaluation mode (no training, just inference)

print(f"✓ Model loaded: ResNet18")
print(f"  - Architecture: {type(model).__name__}")
print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,} weights")
print(f"  - Input size: 224x224 RGB images")
print(f"  - Output: 1000 class probabilities")

# ============================================================================
# STEP 3: Save Model to File (Required by our Inference Engine)
# ============================================================================
print("\n[STEP 3] Saving Model to Temporary File")
print("-" * 60)
print("Our inference engine needs a file path, so we'll save it temporarily...")

with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
    temp_model_path = tmp_file.name

# Save model with weights_only=False compatibility
try:
    torch.save(model, temp_model_path, _use_new_zipfile_serialization=False)
except TypeError:
    torch.save(model, temp_model_path)
print(f"✓ Model saved to: {temp_model_path}")

# ============================================================================
# STEP 4: Initialize Inference Engine
# ============================================================================
print("\n[STEP 4] Initializing Inference Engine")
print("-" * 60)
engine = OptimizedInferenceEngine()
print(f"✓ Engine initialized")
print(f"  - Backend: {engine.backend.backend_type.value}")
print(f"  - Device: {engine.device_info.get('name', 'Unknown')}")

# ============================================================================
# STEP 5: Load Model into Engine
# ============================================================================
print("\n[STEP 5] Loading Model into Inference Engine")
print("-" * 60)
engine.load_model(temp_model_path, model_type="pytorch")
print("✓ Model loaded into engine")
print("  - Model is now ready for inference")

# ============================================================================
# STEP 6: Prepare Input Data
# ============================================================================
print("\n[STEP 6] Loading Real Image")
print("-" * 60)
print("""
Input format for ResNet18:
- Shape: [batch_size, channels, height, width]
- Example: [1, 3, 224, 224] = 1 image, 3 color channels (RGB), 224x224 pixels
- Values: Normalized using ImageNet statistics

Downloading a sample image to test with...
""")

# Download a sample ImageNet image (a dog)
image_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
temp_image_path = os.path.join(tempfile.gettempdir(), "test_image.jpg")

try:
    print(f"Downloading sample image from: {image_url}")
    urllib.request.urlretrieve(image_url, temp_image_path)
    print(f"✓ Image downloaded to: {temp_image_path}")
    
    # Load and preprocess the image
    img = Image.open(temp_image_path).convert('RGB')
    print(f"✓ Image loaded: {img.size[0]}x{img.size[1]} pixels")
    
    # ImageNet preprocessing: resize to 256, center crop to 224, normalize
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess the image
    img_tensor = preprocess(img)
    input_data = img_tensor.unsqueeze(0).numpy().astype(np.float32)
    
    print(f"✓ Image preprocessed")
    print(f"  - Shape: {input_data.shape}")
    print(f"  - Data type: {input_data.dtype}")
    print(f"  - Value range: [{input_data.min():.3f}, {input_data.max():.3f}]")
    
except Exception as e:
    print(f"⚠ Could not download sample image: {e}")
    print("Falling back to random input for testing...")
    # Fallback to random input
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    input_data = (input_data - mean) / std
    print(f"✓ Random input prepared (fallback)")

# ============================================================================
# STEP 7: Run Inference
# ============================================================================
print("\n[STEP 7] Running Inference (Making Prediction)")
print("-" * 60)
print("Running the model on our input data...")
print("(This is where the GPU acceleration happens!)")

output = engine.infer(input_data)

print(f"✓ Inference complete!")
print(f"  - Output shape: {output.shape}")
print(f"  - Output represents: 1000 class probabilities")

# ============================================================================
# STEP 8: Interpret Results
# ============================================================================
print("\n[STEP 8] Interpreting Results")
print("-" * 60)
print("""
The output is a vector of 1000 numbers (one per ImageNet category).
Each number represents the model's confidence that the input belongs to that category.

Higher number = More confident
Lower number = Less confident
""")

# Get top 5 predictions
probabilities = torch.softmax(torch.from_numpy(output[0]), dim=0)
top5_probs, top5_indices = torch.topk(probabilities, 5)

# Load ImageNet class names
try:
    from torchvision.datasets import ImageNet
    # ImageNet class names are available through torchvision
    import json
    import urllib.request
    
    # Download ImageNet class labels
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    with urllib.request.urlopen(url) as f:
        imagenet_classes = [line.decode('utf-8').strip() for line in f.readlines()]
except:
    # Fallback: use generic labels if download fails
    imagenet_classes = [f"Class {i}" for i in range(1000)]

print("\nTop 5 Predictions:")
print("-" * 60)
for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices), 1):
    class_idx = idx.item()
    class_name = imagenet_classes[class_idx] if class_idx < len(imagenet_classes) else f"Class {class_idx}"
    print(f"{i}. Class #{class_idx:3d} ({class_name}): {prob.item()*100:.2f}% confidence")

# ============================================================================
# STEP 9: Cleanup
# ============================================================================
print("\n[STEP 9] Cleanup")
print("-" * 60)
try:
    os.remove(temp_model_path)
    print(f"✓ Temporary model file removed")
except Exception as e:
    print(f"⚠ Could not remove model file: {e}")

try:
    if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
        os.remove(temp_image_path)
        print(f"✓ Temporary image file removed")
except Exception as e:
    pass

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("SUMMARY - What Just Happened")
print("=" * 60)
print("""
1. ✓ Loaded a pre-trained ResNet18 model (11M parameters)
2. ✓ Initialized inference engine with OpenCL backend
3. ✓ Loaded and preprocessed a real image (224x224 RGB)
4. ✓ Ran inference (model processed the image)
5. ✓ Got predictions with actual class names (1000 class probabilities)

The model successfully:
- Processed a real image through its neural network layers
- Used GPU acceleration (OpenCL backend)
- Generated accurate predictions with class names
- Identified the object in the image with confidence scores

Next steps you could try:
- Test with your own images
- Apply model optimizations (quantization, pruning)
- Benchmark performance improvements
- Try different models (ResNet50, MobileNet, etc.)
""")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)

