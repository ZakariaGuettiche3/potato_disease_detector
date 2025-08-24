from fastapi import FastAPI , File , UploadFile , HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from io import BytesIO
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
TRASNFORM = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256, 256)),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                     std=[0.5, 0.5, 0.5])])
                                            
All_CLASSES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

class CNNModele(nn.Module):
    def __init__(self, num_classes):
        super(CNNModele, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 2 * 2, 128)  
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.relu1(self.conv1(x)) 
        out = self.pool(out)      
        out = self.relu2(self.conv2(out)) 
        out = self.pool(out)   
        out = self.relu3(self.conv3(out)) 
        out = self.pool(out) 
        out = self.relu4(self.conv4(out)) 
        out = self.pool(out) 
        out = self.relu5(self.conv5(out)) 
        out = self.pool(out)
        out = self.relu6(self.conv6(out)) 
        out = self.pool(out)
        out = self.relu7(self.conv7(out)) 
        out = self.pool(out)
        out = out.view(out.size(0), -1)  
        out = self.fc1(out)
        out = self.fc2(out)
        return out

MODEL = CNNModele(3)
MODEL.load_state_dict(torch.load("C:/Users/DELL/Desktop/datata/model_V2.pth", map_location="cpu"))
MODEL.eval()

async def read_image_as_tensor(file: UploadFile) -> np.ndarray:
    image = TRASNFORM(Image.open(BytesIO(file.file.read())).convert('RGB')).unsqueeze(0)
    return image

@app.get('/ping')
async def ping():
    return 'Hi !!!!!'

@app.post('/predict')
async def predect(file : UploadFile = File(...)):
    allowed_extensions = {"jpg", "jpeg", "png", "gif"}
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file_extension}. Only {allowed_extensions} allowed."
        )
    image = await read_image_as_tensor(file=file)
    with torch.no_grad():
        output = MODEL(image)
        probs = F.softmax(output, dim=1)
        confidence,predicted = torch.max(probs, 1)
    return JSONResponse({
        'class': All_CLASSES[int(predicted.item())],
        'confidence': round(confidence.item(), 4)*100
    })
    
 

if __name__ == "__main__":
    uvicorn.run(app , host='localhost' , port=8001)