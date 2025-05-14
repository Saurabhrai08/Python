# Skin Cancer Detection using CNN

This project implements a Convolutional Neural Network (CNN) for binary classification of skin lesions into benign and malignant categories. The model is built using TensorFlow and Keras, with image preprocessing, training, evaluation, and visualization all included in the workflow.

Dataset
The dataset is provided as a ZIP file (skin_cancer_detect.zip) containing two folders corresponding to the two classes: benign and malignant. The images are resized to 256x256 pixels, and an 80/20 split is applied for training and validation using ImageDataGenerator.

  --Training images: 2,638

  --Validation images: 659

  --Image size: 256 x 256

  --Classes: Benign, Malignant

Model Architecture
The CNN model is created using Keras' Sequential API and includes:

Convolutional layers with ReLU activation

MaxPooling layers to reduce spatial dimensions

Flatten layer for converting features to 1D

Fully connected Dense layer with Dropout

Output layer with softmax activation for binary classification

Model details:

Optimizer: Adam

Loss function: Categorical Crossentropy

Metrics: Accuracy

Total parameters: ~14.8 million

Training
The model is trained for up to 20 epochs with early stopping based on validation loss to prevent overfitting.

Best validation accuracy: Approximately 85%

Batch size: 32

Epoch duration: Around 6–7 seconds on a Colab GPU

Evaluation
After training, the model is evaluated using:

Confusion matrix

Classification report (precision, recall, F1-score)

Accuracy and loss plots over epochs

Results on the validation set:

Accuracy: 85%

Benign class: Precision = 89%, Recall = 82%

Malignant class: Precision = 81%, Recall = 88%

Model Export
The trained model is saved in HDF5 format as a1_model.h5 and can be downloaded or reused for deployment and inference tasks.

Streamlit Web Application
A Streamlit app has been developed to serve the trained model through a simple and interactive user interface. Users can upload a skin lesion image, and the app will predict whether the lesion is benign or malignant using the trained CNN model.

To run the app:

bash
Copy
Edit
streamlit run app.py
This makes it easy for users to test the model in real time without writing code.

Visualizations
Training and validation accuracy/loss plots

Confusion matrix heatmap using seaborn

Getting Started
Upload skin_cancer_detect.zip to your working directory.

Run the script or notebook to extract, preprocess, train, and evaluate the model.

After training, the model file a1_model.h5 will be saved in the directory.

To use the Streamlit interface, run the app script.
