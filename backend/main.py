import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
from typing import List
from pathlib import Path
from fastapi.responses import JSONResponse

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
model = SimpleSegNet(num_classes=2).to(device)
model.load_state_dict(torch.load("segnet_model.pth"))  # Load a pretrained model if available
model.eval()

# Helper function to run the model on the image
def run_segnet_on_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("L")  # Convert to grayscale
    image = torch.tensor(np.array(image)).unsqueeze(0).unsqueeze(0).float().to(device)  # Add batch and channel dims
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
                result_data = {
                    'image_name': image_name,
                    'segmentation_result': seg_result.cpu().numpy().tolist()  # Convert to list for JSON serialization
                }
                results.append(result_data)
        return results

# Route to handle file upload and SegNet processing
@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    try:
        results = process_zip_file(file)
        return JSONResponse(content={"status": "success", "results": results}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
