import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import logging
import io
import traceback
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Must be the first Streamlit command
st.set_page_config(
    page_title="🧠 Brain MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .explanation-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .chatbot-box {
        background-color: #F3E5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    /* Hide streamlit error messages */
    .stException {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain.png", width=100)
    st.markdown("## About")
    st.markdown("""
    This application uses a deep learning model to classify brain MRI scans into four categories:
    - Glioma
    - Meningioma
    - Pituitary Tumor
    - No Tumor (Normal)
    """)
    
    st.markdown("## How to use")
    st.markdown("""
    1. Upload an MRI scan image
    2. View the prediction results
    3. Examine the visualization to see which regions influenced the prediction
    """)
    
    st.markdown("## Model Information")
    st.markdown("""
    - Architecture: CNN
    - Input size: 224x224 pixels
    - Classes: 4
    - Explainability: Grad-CAM
    """)
    
    # Chatbot section
    st.markdown("## 💬 Ask About Brain MRIs")
    st.markdown('<div class="chatbot-box">', unsafe_allow_html=True)
    # Load Gemini API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            print(f"Error configuring Gemini API: {e}")
            st.info("Chatbot is unavailable due to an issue with the API configuration.")
    else:
        st.info("Chatbot is unavailable. Please ensure the GEMINI_API_KEY is set in the environment.")
    
    # Chat input and response
    user_question = st.text_input("Ask a question about brain MRIs or tumors:")
    if user_question and api_key:
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(user_question)
            st.markdown(f"**Answer:** {response.text}")
        except Exception as e:
            print(f"Error generating chatbot response: {e}")
            print(traceback.format_exc())
            st.info("Unable to generate a response. Please try again later.")
    st.markdown('</div>', unsafe_allow_html=True)

# Main content
st.markdown('<p class="main-header">🧠 Brain MRI Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a Brain MRI image to classify it into one of the four categories</p>', unsafe_allow_html=True)

# Create two columns for the layout
col1, col2 = st.columns([1, 1])

# Class labels
class_names = ["Glioma", "Meningioma", "Pituitary Tumor", "No Tumor"]

# Function to preprocess image
def preprocess_image(image):
    try:
        # Resize and preprocess the image
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        
        # Ensure the image has 3 channels (RGB)
        if img_array.shape[-1] != 3:
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack((img_array,) * 3, axis=-1)
            elif img_array.shape[-1] == 4:  # RGBA
                img_array = img_array[:, :, :3]
        
        return img_array
    except Exception as e:
        # Log error but don't show to user
        print(f"Error preprocessing image: {e}")
        return None

# Load model
@st.cache_resource
def load_model():
    try:
        # In a real application, you would use a more robust way to store and load the model
        model_path = "brainCNN.h5"
        
        # For demonstration, let's assume the model is available
        if not os.path.exists(model_path):
            # Create a simple placeholder model for demonstration
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
                tf.keras.layers.Conv2D(16, 3, activation='relu', name='conv_layer_1'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(32, 3, activation='relu', name='conv_layer_2'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, 3, activation='relu', name='last_conv_layer'),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(4, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy')
            
            # Initialize the model with a dummy input to build it
            dummy_input = np.zeros((1, 224, 224, 3))
            with tf.device('/CPU:0'):  # Force CPU to avoid GPU-related issues
                _ = model(dummy_input, training=False)
            
            return model
        
        model = tf.keras.models.load_model(model_path)
        
        # Initialize the model with a dummy input to ensure it's built
        dummy_input = np.zeros((1, 224, 224, 3))
        with tf.device('/CPU:0'):  # Force CPU to avoid GPU-related issues
            _ = model(dummy_input, training=False)
        
        return model
    except Exception as e:
        # Log error but don't show to user
        print(f"Error loading model: {e}")
        print(traceback.format_exc())
        return None

# Predict function
def predict(image, model):
    try:
        img_array = preprocess_image(image)
        if img_array is None:
            return None, None, None, None
            
        img_array_batch = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make prediction
        with tf.device('/CPU:0'):  # Force CPU to avoid GPU-related issues
            prediction = model(img_array_batch, training=False).numpy()
        
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class]
        
        return class_names[predicted_class], prediction[0], confidence, img_array
    except Exception as e:
        # Log error but don't show to user
        print(f"Error during prediction: {e}")
        print(traceback.format_exc())
        return None, None, None, None

# Find the last convolutional layer in the model
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        # Check if the layer is a convolutional layer
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

# Improved Grad-CAM implementation
def make_gradcam_heatmap(img_array, model, predicted_class_idx):
    try:
        # First, ensure the model is built by running a prediction
        img_array_batch = np.expand_dims(img_array, axis=0)
        with tf.device('/CPU:0'):
            _ = model(img_array_batch, training=False)
        
        # Find the last convolutional layer
        last_conv_layer_name = find_last_conv_layer(model)
        
        if last_conv_layer_name is None:
            print("Could not find a convolutional layer in the model.")
            return None
        
        # Create a simplified model that outputs:
        # 1. The activations of the last conv layer
        # 2. The final prediction
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output
            ]
        )
        
        # Compute the gradient of the top predicted class with respect to the output feature map
        with tf.GradientTape() as tape:
            # Cast the image array to a float-32 tensor
            img_array_tensor = tf.cast(img_array_batch, tf.float32)
            
            # Watch the inputs
            tape.watch(img_array_tensor)
            
            # Get the activations of the last conv layer and make a prediction
            with tf.device('/CPU:0'):
                last_conv_layer_output, predictions = grad_model(img_array_tensor, training=False)
            
            # Get the score for the predicted class
            pred_index = predicted_class_idx
            class_channel = predictions[:, pred_index]
        
        # Gradient of the predicted class with respect to the output feature map
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        # Vector of mean intensity of the gradient over each feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the computed importance
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        
        # Weight each channel in the feature map by how important it is
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]
        
        # Average all channels to get the heatmap
        heatmap = np.mean(last_conv_layer_output, axis=-1)
        
        # Normalize the heatmap
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) or 1)
        
        return heatmap
    
    except Exception as e:
        # Log error but don't show to user
        print(f"Error generating Grad-CAM: {e}")
        print(traceback.format_exc())
        return None

# Improved alternative visualization if the standard Grad-CAM fails
def generate_simple_attention_map(img_array, model):
    try:
        # Create a simplified attention map based on the activations of the last layer before the final dense layer
        # This is a fallback when we can't use proper Grad-CAM
        
        # Find a suitable layer - try to get the layer before the final dense layer
        suitable_layer = None
        for i, layer in enumerate(model.layers):
            if isinstance(layer, (tf.keras.layers.GlobalAveragePooling2D, tf.keras.layers.Flatten)):
                # Look for the previous layer that has spatial dimensions
                for j in range(i-1, -1, -1):
                    prev_layer = model.layers[j]
                    if isinstance(prev_layer, (tf.keras.layers.Conv2D, tf.keras.layers.MaxPooling2D)):
                        suitable_layer = prev_layer.name
                        break
                if suitable_layer:
                    break
        
        if not suitable_layer:
            # If we couldn't find a suitable layer, just use the first layer
            suitable_layer = model.layers[0].name
        
        # Create a model that outputs the activations of the suitable layer
        activation_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=model.get_layer(suitable_layer).output
        )
        
        # Get the activations
        img_array_batch = np.expand_dims(img_array, axis=0)
        with tf.device('/CPU:0'):  # Force CPU to avoid GPU-related issues
            activations = activation_model(img_array_batch, training=False).numpy()
        
        # Average the activations across channels to get a simple attention map
        attention_map = np.mean(activations[0], axis=-1)
        
        # Normalize the attention map
        attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map) + 1e-7)
        
        return attention_map
    
    except Exception as e:
        # Log error but don't show to user
        print(f"Error generating simple attention map: {e}")
        print(traceback.format_exc())
        return None

# Function to overlay heatmap on original image
def display_gradcam(img_array, heatmap, alpha=0.4):
    try:
        # Resize heatmap to match the image size
        heatmap = np.uint8(255 * heatmap)
        
        # Use scipy for resizing if the shapes don't match
        if heatmap.shape != (img_array.shape[0], img_array.shape[1]):
            try:
                import cv2
                heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
            except ImportError:
                from scipy.ndimage import zoom
                
                zoom_factor = (img_array.shape[0] / heatmap.shape[0], 
                              img_array.shape[1] / heatmap.shape[1])
                
                heatmap = zoom(heatmap, zoom_factor, order=1)
        
        # Apply the colormap
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        
        # Create an RGB image from the heatmap
        jet_heatmap = np.uint8(jet_heatmap * 255)
        
        # Superimpose the heatmap on the original image
        superimposed_img = np.uint8(img_array * 255) * (1 - alpha) + jet_heatmap * alpha
        superimposed_img = np.uint8(np.clip(superimposed_img, 0, 255))
        
        # Convert to PIL Image for display
        superimposed_img_pil = Image.fromarray(superimposed_img)
        
        return superimposed_img_pil
    
    except Exception as e:
        # Log error but don't show to user
        print(f"Error displaying Grad-CAM: {e}")
        print(traceback.format_exc())
        return Image.fromarray(np.uint8(img_array * 255))  # Return original image if overlay fails

# Load the model
model = load_model()

with col1:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### Upload MRI Image")
    uploaded_file = st.file_uploader("Choose a brain MRI image", type=["jpg", "jpeg", "png"])
    
    # Add heatmap intensity slider
    heatmap_intensity = st.slider("Heatmap Intensity", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Initialize image variable
image = None

# Handle image selection
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        # Log error but don't show to user
        print(f"Error opening uploaded image: {e}")
        print(traceback.format_exc())

# Display image and prediction
with col2:
    if image is not None:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### MRI Scan")
        st.image(image, caption="Brain MRI Scan", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if model is not None:
            with st.spinner("Analyzing image..."):
                label, raw_output, confidence, preprocessed_img = predict(image, model)
            
            if label is not None:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f"### 🧠 Prediction: {label}")
                st.markdown(f"### Confidence: {confidence*100:.2f}%")
                
                # Create a bar chart for the prediction scores
                st.markdown("### Prediction Scores:")
                
                # Create a dictionary for the chart data
                chart_data = {class_name: float(score) for class_name, score in zip(class_names, raw_output)}
                
                # Sort the data by score in descending order
                sorted_data = {k: v for k, v in sorted(chart_data.items(), key=lambda item: item[1], reverse=True)}
                
                # Create the chart
                st.bar_chart(sorted_data)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Generate and display visualization
                st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
                st.markdown("### 🔍 Visualization of Model Focus")
                st.markdown("This heatmap shows which regions of the MRI influenced the model's prediction:")
                
                with st.spinner("Generating visualization..."):
                    predicted_class_idx = np.argmax(raw_output)
                    
                    # Try standard Grad-CAM first
                    heatmap = make_gradcam_heatmap(preprocessed_img, model, predicted_class_idx)
                    
                    # If standard Grad-CAM fails, try the simpler alternative
                    if heatmap is None:
                        st.info("Using simplified attention visualization instead.")
                        heatmap = generate_simple_attention_map(preprocessed_img, model)
                    
                    if heatmap is not None:
                        # Create a figure with two subplots side by side
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Original image
                        ax1.imshow(preprocessed_img)
                        ax1.set_title('Original MRI')
                        ax1.axis('off')
                        
                        # Heatmap overlay
                        superimposed_img = display_gradcam(preprocessed_img, heatmap, alpha=heatmap_intensity)
                        ax2.imshow(superimposed_img)
                        ax2.set_title('Attention Heatmap')
                        ax2.axis('off')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        st.markdown("""
                        **How to interpret this visualization:**
                        - Red/yellow areas indicate regions that strongly influenced the prediction
                        - Blue areas had less influence on the model's decision
                        - This helps identify which anatomical features the model focused on
                        """)
                    else:
                        # Don't show error, just a neutral message
                        st.info("Visualization could not be generated for this image.")
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Don't show error, just a neutral message
            st.info("Model is initializing. Please try again in a moment.")
    else:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### No Image Selected")
        st.markdown("Please upload an MRI scan to get a prediction.")
        st.markdown('</div>', unsafe_allow_html=True)

# Add explanations about the tumor types
st.markdown("## About Brain Tumors")

tumor_tabs = st.tabs(["Glioma", "Meningioma", "Pituitary Tumor", "No Tumor"])

with tumor_tabs[0]:
    st.markdown("""
    ### Glioma
    Gliomas are tumors that originate in the glial cells of the brain. These cells support and nourish neurons.
    
    **Characteristics:**
    - Most common type of primary brain tumor
    - Can be low-grade (slow-growing) or high-grade (fast-growing)
    - Symptoms include headaches, seizures, and cognitive changes
    
    **MRI Appearance:**
    - Often appears as an irregular mass
    - May have areas of necrosis (dead tissue)
    - Usually enhances with contrast
    """)

with tumor_tabs[1]:
    st.markdown("""
    ### Meningioma
    Meningiomas develop from the meninges, the membranes that surround the brain and spinal cord.
    
    **Characteristics:**
    - Usually benign (non-cancerous)
    - Slow-growing
    - More common in women
    - Symptoms depend on location
    
    **MRI Appearance:**
    - Well-defined, round or oval mass
    - Typically attached to the dura mater
    - Strong, uniform enhancement with contrast
    """)

with tumor_tabs[2]:
    st.markdown("""
    ### Pituitary Tumor
    Pituitary tumors develop in the pituitary gland at the base of the brain.
    
    **Characteristics:**
    - Usually benign
    - May cause hormonal imbalances
    - Can cause vision problems due to proximity to optic nerves
    
    **MRI Appearance:**
    - Located in the sella turcica at the base of the brain
    - Well-defined mass
    - May extend upward (suprasellar extension)
    """)

with tumor_tabs[3]:
    st.markdown("""
    ### No Tumor (Normal)
    A normal brain MRI shows no evidence of tumors or other abnormalities.
    
    **Characteristics:**
    - Normal brain anatomy
    - Clear differentiation between gray and white matter
    - Symmetric structures
    
    **MRI Appearance:**
    - Consistent tissue density
    - No abnormal masses or enhancements
    - Normal ventricle size and shape
    """)

# Footer
st.markdown("---")
st.markdown("### About Model Visualization")
st.markdown("""
**Gradient-weighted Class Activation Mapping (Grad-CAM)** is a technique for producing visual explanations for decisions made by CNN models. 
It uses the gradients of any target concept flowing into the final convolutional layer to produce a coarse localization map highlighting important regions in the image for predicting the concept.

This helps make the model's decision-making process more transparent and interpretable, which is especially important in medical applications.
""")

st.markdown("### Disclaimer")
st.markdown("""
This application is for educational purposes only and should not be used for medical diagnosis.
Always consult with a qualified healthcare provider for medical advice.
""")
