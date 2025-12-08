import os
os.environ["TRANSFORMERS_NO_TF"] = "1"   # Disable TensorFlow backend for transformers, use PyTorch instead

from transformers import CLIPProcessor, CLIPModel   # Import CLIP model and processor from Hugging Face
from PIL import Image   # Import Pillow to handle images
import torch            # Import PyTorch for tensor operations

# DRINK CATEGORIES (labels to classify images)
categories = [
    "soda",
    "water",
    "orange juice",
    "coffee",
    "iced tea"
]

def load_model():
    print("Loading pre-trained CLIP model...")   # Inform the user that the model is being loaded
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")   # Load CLIP model
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")   # Load CLIP processor
    return model, processor   # Return both objects for later use

def predict(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")   # Open image and convert to RGB format

    # Prepare inputs: text categories + image
    inputs = processor(
        text=categories,
        images=image,
        return_tensors="pt",   # Return PyTorch tensors
        padding=True
    )

    # Run model inference without gradient calculation (faster, less memory)
    with torch.no_grad():
        output = model(**inputs)

    logits = output.logits_per_image   # Get similarity scores between image and text
    probs = logits.softmax(dim=1)      # Convert scores into probabilities

    index = probs.argmax().item()         # Find index of highest probability
    confidence = probs[0, index].item()   # Get confidence value of prediction

    return categories[index], confidence   # Return predicted category and confidence

def main():
    model, processor = load_model()   # Load model and processor

    folder = "images_to_test"   # Folder containing test images

    # Iterate through all files in the folder
    for file in os.listdir(folder):
        # Only process image files with these extensions
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder, file)   # Build full path to image
            label, conf = predict(model, processor, path)   # Predict category
            print(f"{file}: {label}  (confidence: {conf:.2f})")   # Print result

# Entry point of the program
if __name__ == "__main__":
    main()
