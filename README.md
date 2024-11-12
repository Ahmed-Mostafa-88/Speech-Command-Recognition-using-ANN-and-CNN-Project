Speech Command Recognition using ANN and CNN
============================================

Project Objective
-----------------

The objective of this project is to implement and compare two deep learning models (Artificial Neural Network (ANN) and Convolutional Neural Network (CNN)) for recognizing speech commands from audio data. The models classify audio samples into one of ten spoken commands (e.g., "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go").

Dataset Information and Preprocessing Steps
-------------------------------------------

### Dataset

The dataset used in this project is the **Google Speech Commands** dataset, which consists of one-second-long audio clips of spoken words representing different commands. The commands selected for this project are:

-   **Commands**: "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"

### Preprocessing Steps

1.  **Audio Loading and Resampling**: Audio files are loaded and resampled to a sample rate of 16,000 Hz.
2.  **MFCC Feature Extraction**: Mel-frequency cepstral coefficients (MFCCs) are extracted from the audio to capture key frequency features.
3.  **Data Augmentation**: To increase the diversity of the dataset, noise-based data augmentation is applied.
4.  **Resizing and Normalization**: The MFCCs are resized to a consistent shape and normalized for better model performance.
5.  **Train-Validation Split**: The dataset is split into training and validation sets.

Model Architectures
-------------------

### ANN Model

-   **Layers**: Input Flatten layer, two Dense layers with ReLU activation, Dropout, BatchNormalization, and an output Dense layer with Softmax activation for multiclass classification.
-   **Loss Function**: Categorical Cross-Entropy
-   **Optimizer**: Adam optimizer

### CNN Model

-   **Layers**: Convolutional layers, MaxPooling, Dropout, BatchNormalization, and Dense layers.
-   **Loss Function**: Categorical Cross-Entropy
-   **Optimizer**: Adam optimizer

Both models are trained on the same dataset and evaluated using accuracy metrics, with confusion matrices generated to assess performance across the ten command categories.

Instructions for Running the Code
---------------------------------

1.  **Clone the Repository**:
git clone https://github.com/Ahmed-Mostafa-88/speech-command-recognition.git\
cd speech-command-recognition

1.  **Dataset Setup**:

    -   Download the **Google Speech Commands** dataset from Google Speech Commands Dataset.
    -   Place the dataset in a directory named `data` (or adjust the path accordingly in the notebook).
2.  **Run the Code**:

    -   Open the Jupyter Notebook `Speech Command Recognition using ANN and CNN.ipynb` in a Jupyter environment (like JupyterLab or Google Colab).
    -   The notebook will contain the implementation for both ANN and CNN models, and it will automatically run the training and evaluation of both models.

    The notebook includes:

    -   Code for data preprocessing (loading, feature extraction, augmentation)
    -   ANN and CNN model training and evaluation
    -   Plotting of training and validation accuracy/loss curves
    -   Generation of confusion matrices for both models
3.  **View Results**:

    -   The notebook will display:
        -   Training and validation accuracy/loss plots.
        -   Confusion matrices for the ANN and CNN models.

Dependencies and Installation Instructions
------------------------------------------

### Requirements

-   Python >= 3.7
-   TensorFlow >= 2.0
-   Keras
-   NumPy
-   SciPy
-   Librosa (for audio processing)
-   Matplotlib (for plotting)
-   Seaborn (for confusion matrix visualization)
-   scikit-learn (for confusion matrix and classification report)

### Installation

1.  **Create and Activate a Virtual Environment** (optional but recommended):
python3 -m venv env\
source env/bin/activate  # On Windows use `env\Scripts\activate`
2.  **Install Dependencies**: Install all required dependencies with:
pip install tensorflow keras numpy scipy librosa matplotlib seaborn scikit-learn

File Structure
--------------

-   `Speech Command Recognition using ANN and CNN.ipynb`: Jupyter Notebook containing the full implementation of the project, including data preprocessing, model training, evaluation, and plotting.
-   `README.md`: This README file.

Notes
-----

-   Ensure that the path to the dataset is correctly specified in the notebook.
-   The notebook will display various plots for model performance during training, including accuracy and loss curves.
-   Confusion matrices will be generated for both the ANN and CNN models to analyze classification performance.

License
-------

This project is licensed under the MIT License.
