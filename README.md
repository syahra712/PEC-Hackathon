Here’s a refined and professional version of your README for **TumorVision**, optimized for clarity, structure, and presentation:

---

# 🧠 TumorVision: AI-Powered Brain MRI Analysis with Generative AI

**TumorVision** is an educational web application that utilizes deep learning and generative AI to analyze brain MRI scans. Developed for the **PEC Hackathon**, the app classifies MRIs into four categories—**Glioma**, **Meningioma**, **Pituitary Tumor**, or **No Tumor**—using a custom-built TensorFlow CNN model. It includes Grad-CAM visualizations for model explainability and an AI-powered chatbot for interactive learning.

> ⚠️ **Disclaimer**: TumorVision is intended **for educational purposes only** and should **not be used for medical diagnosis**. Always consult a certified healthcare professional for medical concerns.

---

## 🚀 Features

* **🧠 MRI Classification**
  Upload brain MRI images (JPG, JPEG, PNG) to classify them into one of four categories. Includes confidence scores and a bar chart of class probabilities.

* **🔥 Grad-CAM Visualization**
  Gradient-weighted Class Activation Mapping (Grad-CAM) highlights regions of the MRI that most influenced the model’s prediction. Customize intensity using an interactive slider.

* **💬 Generative AI Chatbot**
  Powered by **Gemini API (gemini-1.5-flash)**, this chatbot answers real-time questions about brain tumors, MRIs, and the application—making it a virtual tutor for learners.

* **📚 Tumor Information Tabs**
  Detailed educational content for each tumor type and normal brain anatomy, including common characteristics and typical MRI appearances.

* **🎨 Streamlit-Based UI**
  Built with Streamlit for a smooth, responsive experience. Includes custom CSS styling and robust error handling for a seamless user journey.

---

## 📹 Demo

Try TumorVision live on [Streamlit Cloud](#) *(Replace this with your deployed URL)*
Or run it locally using the steps below.

---

## 🔧 Installation & Setup

### ✅ Prerequisites

* Python 3.8+
* Git
* Gemini API Key (get it from Google Cloud Console)
* (Optional) A trained brain MRI model (`brainCNN.h5`)

---

### 🛠️ Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/syahra712/pec-hackathon.git
   cd pec-hackathon
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   `requirements.txt` includes:

   * `streamlit`
   * `tensorflow`
   * `numpy`
   * `pillow`
   * `matplotlib`
   * `google-generativeai`

3. **Configure Gemini API Key**

   * Create a `.streamlit/secrets.toml` file:

     ```toml
     GEMINI_API_KEY = "your_valid_gemini_api_key_here"
     ```

   * Or add it via Streamlit Cloud secrets management.

4. **(Optional) Add a Trained Model**

   * Place your trained `brainCNN.h5` in the project root.
   * If absent, a placeholder CNN will be used for demo purposes.

---

## 💻 Running Locally

```bash
streamlit run app-6copy.py
```

Visit [http://localhost:8501](http://localhost:8501) to access the app in your browser.

---

## ☁️ Deployment on Streamlit Cloud

1. Push this repository to your GitHub.
2. Create an account on [Streamlit Cloud](https://streamlit.io/cloud).
3. Link your GitHub repo and deploy the app.
4. Set your `GEMINI_API_KEY` in **Streamlit Secrets**.
5. Specify `app-6copy.py` as the main script.

---

## 🧪 How to Use

1. **Upload an MRI:** Upload a JPG/JPEG/PNG scan in the sidebar.
2. **Get Predictions:** View the predicted class, confidence, and bar chart.
3. **Visualize with Grad-CAM:** Use the slider to adjust heatmap overlay.
4. **Ask the Chatbot:** Get answers to questions like “What is a glioma?”
5. **Learn About Tumors:** Browse the tumor tabs for in-depth descriptions.

---

## ⚙️ Technical Overview

| Component               | Description                                                                 |
| ----------------------- | --------------------------------------------------------------------------- |
| **Frontend**            | Built with Streamlit (custom CSS + responsive layout)                       |
| **CNN Model**           | 3 Conv layers (16, 32, 64), MaxPooling, GAP, Dense (128), Softmax (4-class) |
| **Grad-CAM**            | Uses gradients from the last conv layer, rendered with Matplotlib           |
| **Chatbot**             | Powered by `gemini-1.5-flash`, using Streamlit secrets                      |
| **Image Preprocessing** | Resizes to 224x224, normalizes, converts to RGB (handles grayscale/RGBA)    |
| **Caching**             | `@st.cache_resource` for optimized model loading                            |
| **Error Handling**      | TensorFlow warnings suppressed; Streamlit errors hidden                     |

---

## 📁 Project Structure

```
pec-hackathon/
├── app-6copy.py            # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── secrets.toml        # Gemini API key (not tracked)
├── brainCNN.h5             # Trained CNN model (optional)
└── README.md               # Project documentation
```

---

## 🧑‍💻 Contributing

We welcome contributions to enhance **TumorVision**!

To contribute:

1. Fork this repo.
2. Create a feature branch:

   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:

   ```bash
   git commit -m "Add your feature"
   ```
4. Push to GitHub:

   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request.

### 💡 Potential Enhancements

* Integrate pre-trained CNNs (e.g., ResNet, EfficientNet)
* Add chat history support for the chatbot
* Support multi-sequence MRI scans (T1, T2, FLAIR)
* Include sample images for users to explore

---

## 🛠️ Troubleshooting

| Issue                        | Solution                                                                             |
| ---------------------------- | ------------------------------------------------------------------------------------ |
| **Gemini API Error**         | Ensure `GEMINI_API_KEY` is correctly set and API is enabled in Google Cloud Console. |
| **Model Not Loading**        | Add a valid `brainCNN.h5` file in the root directory.                                |
| **Dependencies Not Working** | Use a virtual environment and reinstall with `pip install -r requirements.txt`.      |
| **API Key Test**             |                                                                                      |

```python
import google.generativeai as genai
genai.configure(api_key="your_key")
model = genai.GenerativeModel('gemini-1.5-flash')
print(model.generate_content("Test").text)
```

---

## 📄 License

This project is licensed under the **MIT License**.
See the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgments

* Built for **PEC Hackathon** by [@syahra712](https://github.com/syahra712)
* Powered by **Streamlit**, **TensorFlow**, and **Google’s Gemini API**
* Inspired by the promise of **Generative AI** in medical education

---

## 📬 Contact

For questions or suggestions, open a GitHub Issue or connect with us during the hackathon!

> **TumorVision**: Where deep learning meets generative AI to illuminate brain MRI analysis.
> Explore, learn, and contribute at [syahra712/pec-hackathon](https://github.com/syahra712/pec-hackathon)!

---

Let me know if you’d like a Markdown-rendered PDF version, or want help deploying on Streamlit Cloud!
