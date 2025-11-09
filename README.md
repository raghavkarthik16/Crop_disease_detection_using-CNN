# Crop_disease_detection_using-CNN
This project implements a Transfer Learning approach using the EfficientNetB4 CNN on the PlantVillage dataset to classify 39 crop disease and health classes. It achieved a robust test accuracy of 93.13%, demonstrating an efficient and highly accurate solution for real-time agricultural diagnostics.

This is an AI-powered web application that can identify various crop diseases from images. The system uses a deep learning model based on EfficientNetB4 architecture to classify plant diseases across different crops.

## Features

- Upload and analyze plant leaf images
- Detect diseases in multiple crop types
- Real-time prediction with confidence scores
- User-friendly web interface
- Supports 39 different plant disease classes

## Prerequisites

- Python 3.10 or higher
- TensorFlow 2.x
- Flask
- Other dependencies (listed in requirements.txt)

## Installation

1. **Clone the Repository**
   ```bash
   git clone [your-repository-url]
   cd Plant-Disease-Recognition-System
   ```

2. **Set Up Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Create Required Directories**
   ```bash
   mkdir -p models uploadimages
   ```

4. **Download the Model**
   - The model will be automatically downloaded when you first run the application
   - It will be saved as `models/crop_disease_detection.keras`

## Usage

1. **Start the Application**
   ```bash
   python app.py
   ```
   The server will start on http://127.0.0.1:5000

2. **Use the Application**
   - Open your web browser and navigate to http://127.0.0.1:5000
   - Upload a plant leaf image through the web interface
   - The system will analyze the image and display:
     - Predicted disease
     - Confidence score
     - Preview of the uploaded image

## Project Structure

```
app.py                  # Main Flask application
plant_disease.json      # Disease class mappings
models/                 # Directory for model files
static/                # Static assets (CSS, JS, images)
templates/             # HTML templates
uploadimages/          # Temporary storage for uploaded images
```

## Technical Details

- Model Architecture: EfficientNetB4
- Input Image Size: 160x160 pixels
- Number of Classes: 39
- Framework: TensorFlow/Keras
- Web Framework: Flask

## Contributing

Feel free to submit issues and enhancement requests!
