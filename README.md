
# Biometric Liveness Detection using EEG Data

This project involves the design and deployment of an Android application to enhance security measures in biometric systems using EEG data. By detecting biometric liveness with state-of-the-art machine learning techniques, this project achieves an impressive **98% accuracy** in detecting liveness.

## Features
- Designed and deployed an Android application for biometric liveness detection.
- Integrated EEG data for biometric verification.
- Used **Generative Adversarial Networks (GANs)**, **Variational Autoencoders (VAEs)**, **Autoencoders**, and **feature extraction** techniques.
- Achieved **98% accuracy** in detecting biometric liveness.

## Tech Stack

| Technology | Description |
|------------|-------------|
| ![Android](https://img.shields.io/badge/Android-3DDC84?style=for-the-badge&logo=android&logoColor=white) | Android SDK for building the mobile app |
| ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) | Python for machine learning model development |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) | TensorFlow for training GANs and Autoencoders |
| ![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white) | Keras for building deep learning models |
| ![GANs](https://img.shields.io/badge/GANs-blue?style=for-the-badge&logo=neural-network&logoColor=white) | Generative Adversarial Networks for generating EEG-based liveness patterns |
| ![VAEs](https://img.shields.io/badge/VAEs-green?style=for-the-badge&logo=neural-network&logoColor=white) | Variational Autoencoders for feature extraction and dimensionality reduction |

## How to Run

1. Clone the repository:
   ```bash
   [git clone https://github.com/your-repo/brain-signal-liveness-detection.git](https://github.com/sakshisingh0598/BrainSignalsVerification_Server.git)
   cd brain-signal-liveness-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the machine learning models:
   ```bash
   python train_model.py
   ```

4. Deploy the Android application:
   - Open the `Android` folder in Android Studio.
   - Build and run the application on your device.

## Contributing

Feel free to submit issues and pull requests to improve the project!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
