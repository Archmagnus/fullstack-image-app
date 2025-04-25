import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
from typing import List
from pathlib import Path
from fastapi.responses import JSONResponse, StreamingResponse
import io

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple SegNet Model
class SimpleSegNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleSegNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, num_classes, 2, stride=2),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# FastAPI app setup
app = FastAPI()

# Initialize the model
MODEL_PATH = "segnet_model.pth"
if os.path.exists(MODEL_PATH):
    model = SimpleSegNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))  # Load a pretrained model if available
    model.eval()
else:
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

# Helper function to run the model on the image
def run_segnet_on_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("L")  # Convert to grayscale
    image = np.array(image)  # Convert to NumPy array
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(device)  # Add batch and channel dims
    with torch.no_grad():
        output = model(image)
    return output

# Helper function to process a zip file
def process_zip_file(zip_file: UploadFile) -> List[dict]:
    with zipfile.ZipFile(zip_file.file, 'r') as zip_ref:
        image_files = zip_ref.namelist()
        results = []
        for image_name in image_files:
            with zip_ref.open(image_name) as file:
                img = Image.open(file)
                seg_result = run_segnet_on_image(img)
                # Convert segmentation result tensor to image for visualization
                seg_image = seg_result.squeeze().cpu().numpy()
                seg_image = np.uint8(seg_image * 255)  # Scaling output to 255 for visualization
                seg_image_pil = Image.fromarray(seg_image)
                
                # Save or convert the result for visualization
                buffered = io.BytesIO()
                seg_image_pil.save(buffered, format="PNG")
                result_data = {
                    'image_name': image_name,
                    'segmentation_result': seg_image_pil  # Save the segmented image to return later
                }
                results.append(result_data)
        return results

# Route to handle file upload and SegNet processing
@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    try:
        results = process_zip_file(file)
        # Return the results
        result_data = []
        for result in results:
            # Convert PIL Image to PNG for response
            buffered = io.BytesIO()
            result['segmentation_result'].save(buffered, format="PNG")
            buffered.seek(0)
            result_data.append({
                'image_name': result['image_name'],
                'segmentation_result': buffered.getvalue()  # return image as binary data
            })
        return JSONResponse(content={"status": "success", "results": result_data}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

