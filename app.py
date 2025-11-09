from flask import Flask, render_template,request,redirect,send_from_directory,url_for
import numpy as np
import json
import uuid
import tensorflow as tf

# Define the labels first
label = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

# Initialize Flask app and load model
app = Flask(__name__)

# Load the model with custom objects
try:
    # Load base model
    base_model = tf.keras.applications.EfficientNetB4(
        include_top=False,
        input_shape=(160, 160, 3),
        weights='imagenet'
    )
    
    # Create the complete model
    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(39, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    
    # Load trained weights
    model.load_weights("models/crop_disease_detection.keras")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Print model information
print("\nModel Information:")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")
print(f"Number of classes: {len(label)}")
print(f"Model summary:")
model.summary()
assert model.output_shape[1] == len(label), "Model output doesn't match number of classes"

# Load disease information
with open("plant_disease.json",'r') as file:
    plant_disease = json.load(file)

# print(plant_disease[4])

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/',methods = ['GET'])
def home():
    return render_template('home.html')

def extract_features(image):
    try:
        # Load and resize image
        image = tf.keras.utils.load_img(image, target_size=(160, 160))
        # Convert to array
        feature = tf.keras.utils.img_to_array(image)
        
        # Preprocess input (normalize using ImageNet mean and std)
        feature = tf.keras.applications.efficientnet.preprocess_input(feature)
        
        # Add batch dimension
        feature = np.array([feature])
        
        # Debug information
        print(f"\nImage preprocessing info:")
        print(f"Shape: {feature.shape}")
        print(f"Value range: [{feature.min():.2f}, {feature.max():.2f}]")
        print(f"Mean value: {feature.mean():.2f}")
        print(f"Std deviation: {feature.std():.2f}")
        
        return feature
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        raise

def model_predict(image):
    try:
        # Extract features from image
        img = extract_features(image)
        
        # Get predictions (softmax is already in the model's last layer)
        predictions = model.predict(img, verbose=0)[0]
        
        # Get top 3 predictions
        top_3_indices = (-predictions).argsort()[:3]
        predicted_index = top_3_indices[0]
        
        print("\nDebug Information:")
        print(f"Top predictions with confidence:")
        for idx in top_3_indices:
            print(f"{label[idx]}: {predictions[idx]*100:.2f}%")
        
        # Get the corresponding disease info
        disease_info = plant_disease[predicted_index].copy()  # Make a copy to avoid modifying original
        disease_info['confidence'] = f"{predictions[predicted_index]*100:.2f}%"
        
        return disease_info
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise
    
    # Get the corresponding disease info from plant_disease JSON
    disease_info = plant_disease[predicted_index]
    disease_info['confidence'] = f"{probabilities[predicted_index]*100:.2f}%"
    return disease_info

@app.route('/upload/',methods = ['POST','GET'])
def uploadimage():
    if request.method == "POST":
        try:
            # Check if file was uploaded
            if 'img' not in request.files:
                return render_template('home.html', error="No file uploaded")
            
            image = request.files['img']
            if image.filename == '':
                return render_template('home.html', error="No file selected")
            
            # Check file type
            if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                return render_template('home.html', error="Invalid file type. Please upload an image file.")
            
            # Save and process image
            temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
            image_path = f'{temp_name}_{image.filename}'
            image.save(image_path)
            print(f'Processing image: {image_path}')
            
            # Get prediction
            disease_info = model_predict(f'./{image_path}')
            
            return render_template('home.html',
                                result=True,
                                imagepath=f'/{image_path}',
                                name=disease_info['name'],
                                cause=disease_info['cause'],
                                cure=disease_info['cure'],
                                confidence=disease_info['confidence'])
        except Exception as e:
            print(f"Error processing image: {e}")
            return render_template('home.html', error=f"Error processing image: {str(e)}")
    
    else:
        return redirect('/')
        
    
if __name__ == "__main__":
    app.run(debug=True)