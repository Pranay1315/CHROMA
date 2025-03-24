import torch
import numpy as np
from PIL import Image
import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from model1 import Generator  # Import your trained generator model

app = Flask(__name__)
CORS(app)

output_dir = "static"
os.makedirs(output_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained generator model
generator = Generator().to(device)
checkpoint = torch.load("generator_latest.pth", map_location=device)
generator.load_state_dict(checkpoint['model_state_dict'])
generator.eval()

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((256, 256))  # Correct PIL resizing
    image = np.array(image) / 255.0 * 2 - 1  # Normalize to [-1, 1]
    image = np.transpose(image, (2, 0, 1))  # Change shape to (C, H, W)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
    return image

@app.route("/", methods=["GET"])
def home():
    return """<h1>Welcome to CHROMA: Image Colorizer</h1>
              <p>Use the <b>/colorize</b> endpoint to upload a black-and-white image and receive a colorized version.</p>"""

@app.route("/colorize", methods=["POST"])
def colorize_image():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["image"]
    try:
        image = Image.open(file).convert("RGB")  # Open as RGB image
    except Exception as e:
        return jsonify({"error": f"Invalid image format: {e}"}), 400

    # Store the original image size
    original_width, original_height = image.size

    # Preprocess image
    input_image = preprocess_image(image)

    # Generate colorized image
    with torch.no_grad():
        colorized_image = generator(input_image)

    # Convert tensor to numpy image
    colorized_image = colorized_image.squeeze(0).cpu().numpy()
    colorized_image = np.transpose(colorized_image, (1, 2, 0))  # Change back to (H, W, C)
    colorized_image = ((colorized_image + 1) / 2 * 255).astype(np.uint8)  # Denormalize to [0, 255]

    # Resize back to the original image size
    colorized_image = Image.fromarray(colorized_image).resize((original_width, original_height))

    # Save and return the colorized image
    output_path = os.path.join(output_dir, "colorized.png")
    colorized_image.save(output_path)  # Save using PIL
    return send_file(output_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
