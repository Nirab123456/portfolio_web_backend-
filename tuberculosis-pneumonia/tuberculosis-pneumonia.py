import os
import json
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from tornado.web import Application, RequestHandler
from tornado.ioloop import IOLoop

# Device setup for CUDA or CPU
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model once at the start
def load_model():
    """Load and initialize the ResNet model for 3-class classification."""
    model = models.resnet18(pretrained=True)  # Load pretrained weights
    model.fc = nn.Linear(model.fc.in_features, 4)  # Modify for 3-class classification: NORMAL, PNEUMONIA, UNKNOWN
    
    # Path to the model file (update the path as necessary)
    model_path = os.path.join('model_22.pt')
    model.load_state_dict(torch.load(model_path, map_location=Device))
    model.to(Device)
    model.eval()
    return model

# Load the model globally
MODEL = load_model()

# Define image transformations
TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet
])

def get_prediction(image_file):
    """Predict if the image indicates NORMAL, PNEUMONIA, or UNKNOWN."""
    try:
        image = Image.open(image_file).convert("RGB")
        transformed_image = TRANSFORMS(image).unsqueeze(0).to(Device)
        with torch.no_grad():
            output = MODEL(transformed_image)
            print('output:', output)
            top_class = output.argmax(dim=1).item()
        # Map numeric predictions to class names
        class_mapping = {0: 'NORMAL', 1: 'PNEUMONIA', 2: 'UNKNOWN', 3: 'TUBERCULOSIS'}
        return class_mapping[top_class]
    except Exception as e:
        return f"Error during prediction: {str(e)}"

class BaseHandler(RequestHandler):
    def set_default_headers(self):
        allowed_origins = ["http://rimajumder.ca","https://rimajumder.ca","rimajumder.ca","http://localhost:3000"]
        origin = self.request.headers.get("Origin", "")  # Get the Origin of the request

        if origin in allowed_origins:
            self.set_header("Access-Control-Allow-Origin", origin)  # Allow only specific origins
        else:
            self.set_header("Access-Control-Allow-Origin", "null")  # Deny all other origins

        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def options(self):
        self.set_status(204)
        self.finish()

# Tornado handler for predictions
class PredictionHandler(BaseHandler):
    async def post(self):
        try:
            # Retrieve uploaded image
            file_body = self.request.files['image'][0]['body']
            
            # Save the uploaded image temporarily
            temp_image_path = "temp_image.jpg"
            with open(temp_image_path, 'wb') as temp_image:
                temp_image.write(file_body)
            
            # Get the prediction
            prediction = get_prediction(temp_image_path)
            
            # Clean up temporary file
            os.remove(temp_image_path)
            
            # Send response
            self.write(json.dumps({'phenomonia_prediction': prediction}))
        except Exception as e:
            self.set_status(500)
            self.write(json.dumps({'error': 'Failed to process the image', 'details': str(e)}))

# Tornado Application setup
def make_app():
    """Create the Tornado application."""
    return Application([
        (r"/predict", PredictionHandler),
    ])

if __name__ == "__main__":
    app = make_app()

    # Fetch PORT from environment variables, default to 8888 if not set
    port = int(os.getenv("PORT", 8888))

    # Fetch the address from environment variables, default to "0.0.0.0" (for Docker)
    address = os.getenv("ADDRESS", "0.0.0.0")

    print(f"Server running on {address}:{port}")
    app.listen(port, address)
    IOLoop.current().start()
