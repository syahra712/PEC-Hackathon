TumorVision: AI-Powered Brain MRI Analysis with Generative AI

TumorVision is an educational web application that leverages deep learning and generative AI to analyze brain MRI scans. Built for the PEC Hackathon, it classifies MRIs into four categories—Glioma, Meningioma, Pituitary Tumor, or No Tumor—using a TensorFlow CNN. The app features Grad-CAM visualizations for explainable AI, a Gemini API-powered chatbot for interactive Q&A, and detailed tumor information tabs, making it a valuable tool for medical students, researchers, and AI enthusiasts.
Disclaimer: TumorVision is for educational purposes only and should not be used for medical diagnosis. Always consult a qualified healthcare provider for medical advice.
Features

MRI Classification: Upload a brain MRI (JPG, JPEG, PNG) to classify it into Glioma, Meningioma, Pituitary Tumor, or No Tumor, with confidence scores and a bar chart of class probabilities.
Grad-CAM Visualization: Uses Gradient-weighted Class Activation Mapping to generate heatmaps highlighting regions influencing the model’s prediction, with an adjustable intensity slider for customizable overlays.
Generative AI Chatbot: Powered by the Gemini API (gemini-1.5-flash), answers questions about brain MRIs, tumors, or the app in real time, acting as an on-demand educational tutor.
Tumor Information Tabs: Provides detailed descriptions of each tumor type and normal brain anatomy, including characteristics and MRI appearances.
User-Friendly Interface: Built with Streamlit, featuring a responsive layout, custom CSS styling, and silent error handling for a seamless experience.

Demo
Check out TumorVision in action on Streamlit Cloud (replace with your deployed URL) or run it locally to explore its features.
Installation
Prerequisites

Python 3.8+
Git
A valid Gemini API key from Google Cloud Console
A brain MRI classification model (brainCNN.h5, optional; the app includes a placeholder CNN if not provided)

Setup

Clone the Repository:
git clone https://github.com/syahra712/pec-hackathon.git
cd pec-hackathon


Install Dependencies:
pip install -r requirements.txt

The requirements.txt includes:
streamlit
tensorflow
numpy
pillow
matplotlib
google-generativeai


Configure the Gemini API Key:

Create a .streamlit/secrets.toml file in the project root:GEMINI_API_KEY = "your_valid_gemini_api_key_here"


Alternatively, set the key in Streamlit Cloud’s secrets management for cloud deployment.
Obtain a key from Google Cloud Console, ensuring the Generative Language API is enabled.


(Optional) Add Trained Model:

Place your trained brainCNN.h5 model in the project root. If absent, the app uses a placeholder CNN for demonstration.



Running Locally
streamlit run app-6copy.py

Open http://localhost:8501 in your browser to view the app.
Deploying on Streamlit Cloud

Push the repository to GitHub.
Create a Streamlit Cloud account and link your repository.
Set GEMINI_API_KEY in Streamlit Cloud’s secrets settings.
Deploy the app, specifying app-6copy.py as the main script.

Usage

Upload an MRI: In the left column, upload a brain MRI image (JPG, JPEG, PNG).
View Prediction: The right column displays the predicted class (e.g., “Glioma”), confidence score, and a bar chart of class probabilities.
Explore Grad-CAM: Adjust the heatmap intensity slider to see which regions influenced the prediction, with red/yellow areas indicating high influence and blue areas low influence.
Ask the Chatbot: In the sidebar, enter questions like “What is a pituitary tumor?” to get instant, AI-generated answers.
Learn About Tumors: Navigate to the “About Brain Tumors” tabs for detailed information on each class.

Technical Details

Framework: Streamlit for the front-end, TensorFlow for the CNN, Google Gemini API for the chatbot.
Model Architecture: A CNN with three convolutional layers (16, 32, 64 filters), max-pooling, global average pooling, and dense layers (128 units, 4-class softmax output).
Grad-CAM: Generates heatmaps using gradients from the last convolutional layer, resized with OpenCV or SciPy, and visualized with Matplotlib.
Chatbot: Uses gemini-1.5-flash for natural language responses, configured via Streamlit secrets for secure API access.
Input Processing: Resizes images to 224x224 pixels, normalizes to [0,1], and converts to RGB (handles grayscale/RGBA).
Error Handling: Suppresses TensorFlow warnings, hides Streamlit exceptions, and logs errors silently to the console.
Deployment: Supports local and cloud deployment, with @st.cache_resource for optimized model loading and CPU-based inference.

Project Structure
pec-hackathon/
├── app-6copy.py              # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── secrets.toml          # Gemini API key (not tracked)
├── brainCNN.h5               # Trained model (optional)
└── README.md                 # This file

Contributing
We welcome contributions to enhance TumorVision! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Potential enhancements:

Integrate pre-trained models (e.g., ResNet) for improved accuracy.
Add chat history to the chatbot for context-aware responses.
Support multi-sequence MRI analysis (e.g., T1, T2, FLAIR).
Include sample MRI images for demo purposes.

Troubleshooting

Gemini API Key Error: Ensure GEMINI_API_KEY is set in .streamlit/secrets.toml or Streamlit Cloud secrets. Verify the key in Google Cloud Console and enable the Generative Language API.
Model Loading: If brainCNN.h5 is missing, the app uses a placeholder CNN. Place a trained model in the root directory.
Dependencies: Run pip install -r requirements.txt in a virtual environment to avoid conflicts.
Logs: Check console logs for errors (e.g., “GEMINI_API_KEY not found”). Test the API key with:import google.generativeai as genai
genai.configure(api_key="your_key")
model = genai.GenerativeModel('gemini-1.5-flash')
print(model.generate_content("Test").text)



License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Built for the PEC Hackathon by syahra712.
Powered by Streamlit, TensorFlow, and Google’s Gemini API.
Inspired by the potential of GenAI to advance healthcare education.

Contact
For questions or feedback, reach out via GitHub Issues or connect with us at the hackathon!

TumorVision: Where deep learning meets generative AI to illuminate brain MRI analysis. Explore, learn, and contribute at syahra712/pec-hackathon!
