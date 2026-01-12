# ❤️ Heart Sound Classification System

A machine learning-based application for classifying heart sounds as **Normal** or **Abnormal** using Phonocardiogram (PCG) recordings.

## 🚀 Overview

This project uses advanced signal processing and gradient-boosted decision trees to analyze heart sounds. It features a high-performance training pipeline and a user-friendly Streamlit web interface for real-time classification.

## 🛠️ How It Works

The system follows a standard machine learning pipeline tailored for audio analysis:

1.  **Audio Ingestion**: Heart sound recordings (.wav) are loaded and standardized to a 22,050 Hz sampling rate.
2.  **Feature Extraction**: Complex audio signals are converted into 60-dimensional feature vectors capturing timbre, frequency content, and temporal dynamics.
3.  **Scaling**: Features are normalized using a `StandardScaler` to ensure all metrics contribute equally to the model decision.
4.  **Inference**: A trained **Gradient Boosting Classifier** analyzes the features to provide a classification and confidence score.

## 📊 Feature Extraction Techniques

We extract a rich set of 60 features per audio clip to ensure the model captures the subtle nuances of heart murmurs:

- **MFCCs (Mel-Frequency Cepstral Coefficients)**: 13 coefficients (mean & variance) that represent the "timbre" or texture of the sound.
- **Delta-MFCCs**: 13 coefficients (mean & variance) that capture the first-order derivative of MFCCs, representing how the sound changes over time.
- **Spectral Centroid**: Measures where the "center of mass" of the spectrum is located (indicates brightness).
- **Spectral Bandwidth**: Measures the width of the spectral distribution.
- **Spectral Rolloff**: The frequency below which a specified percentage (85%) of the total spectral energy lies.
- **Zero Crossing Rate (ZCR)**: The rate at which the signal changes sign—helping to identify noise levels often associated with specific murmurs.

## 🧠 Model Training

The model is built using the following techniques to maximize accuracy and handle class imbalance:

- **Algorithm**: `GradientBoostingClassifier`. We chose this over Random Forests because it performs better at refining the decision boundary in complex medical datasets.
- **Optimization**:
  - **Parallel Processing**: Uses `joblib` to parallelize feature extraction across all CPU cores, reducing training time from ~30 minutes to under 1 minute.
  - **Caching**: Extracted features are cached as `.npy` files to allow for rapid iterative training without re-processing audio.
- **Preprocessing**: Uses `StandardScaler` trained on the training distribution to ensure inference matches training conditions.

## 💻 Tech Stack

- **Logic**: Python 3.x
- **Package Manager**: `uv` (Fastest Python package manager)
- **Audio Processing**: `librosa`
- **Machine Learning**: `scikit-learn`
- **Frontend**: `Streamlit`
- **Persistence**: `joblib`

## 🏃 Getting Started

### Prerequisites

Ensure you have `uv` installed. If not, install it via:

```pip
pip install uv
```

### Installation

1. Clone the repository.
2. Install dependencies:
   ```pip
   uv sync
   ```

### Training the Model

To re-train the model with the latest features:

```bash
uv run model_training.py
```

### Running the App

To start the web interface:

```bash
uv run streamlit run app.py
```

## � Project Structure

```
mutimodel-heart-sounds/
├── app.py                    # Streamlit web interface
├── model_training.py         # Model training pipeline
├── main.py                   # Main entry point
├── check_distribution.py     # Data distribution analysis
├── pyproject.toml            # Project dependencies (uv)
├── README.md                 # Project documentation
├── X_features.npy            # Cached feature vectors (not in repo)
├── y_labels.npy              # Cached labels (not in repo)
├── heart_sound_models/       # Trained models (not in repo)
│   ├── heart_sound_model.pkl
│   ├── label_encoder.pkl
│   └── scaler.pkl
├── pcg_data1/                # Dataset 1 (not in repo)
└── pcg_data2/                # Dataset 2 (not in repo)
```

## 🎯 Usage Example

1. Place your heart sound recordings in `.wav` format
2. Launch the web interface: `uv run streamlit run app.py`
3. Upload your audio file through the web interface
4. View the classification result and confidence score

## 📈 Current Performance

- **Accuracy**: ~83%
- **Class Imbalance**: The model uses balanced strategy to handle the scarcity of "Abnormal" heart sound records in the dataset.

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Submit pull requests

## 📝 License

This project is open source and available for educational and research purposes.

## 📧 Contact

For questions or collaboration opportunities, please open an issue in the repository.

---

**Note**: This project is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis.
