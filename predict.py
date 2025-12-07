import os
os.environ["TRANSFORMERS_NO_TF"] = "1"   # Disable TensorFlow backend for transformers, use PyTorch instead
from transformers import CLIPProcessor, CLIPModel   # Import CLIP model and processor from Hugging Face
from PIL import Image   # Import Pillow to handle images
import torch            # Import PyTorch for tensor operations
import os               # Import os again (already imported above, but harmless)

# DRINK CATEGORIES (labels to classify images)
categorias = [
"Coca-Cola", 
"Pepsi", 
"water", 
"orange juice", 
"coffee", 
"iced tea"
]

def cargar_modelo():
    print("Loading pre-trained CLIP model...")   # Inform the user that the model is being loaded
    modelo = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")   # Load CLIP model
    procesador = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")   # Load CLIP processor
    return modelo, procesador   # Return both objects for later use

def predecir(modelo, procesador, ruta_imagen):
    imagen = Image.open(ruta_imagen).convert("RGB")   # Open image and convert to RGB format

    # Prepare inputs: text categories + image
    entradas = procesador(
        text=categorias,
        images=imagen,
        return_tensors="pt",   # Return PyTorch tensors
        padding=True
    )

    # Run model inference without gradient calculation (faster, less memory)
    with torch.no_grad():
        salida = modelo(**entradas)

    logits = salida.logits_per_image   # Get similarity scores between image and text
    probs = logits.softmax(dim=1)      # Convert scores into probabilities

    indice = probs.argmax().item()     # Find index of highest probability
    confianza = probs[0, indice].item()   # Get confidence value of prediction

    return categorias[indice], confianza   # Return predicted category and confidence

def main():
    modelo, procesador = cargar_modelo()   # Load model and processor

    carpeta = "imagenes_para_probar"   # Folder containing test images

    # Iterate through all files in the folder
    for archivo in os.listdir(carpeta):
        # Only process image files with these extensions
        if archivo.lower().endswith((".jpg", ".png", ".jpeg")):
            ruta = os.path.join(carpeta, archivo)   # Build full path to image
            etiqueta, conf = predecir(modelo, procesador, ruta)   # Predict category
            print(f"{archivo}: {etiqueta}  (confidence: {conf:.2f})")   # Print result

# Entry point of the program
if __name__ == "__main__":
    main()