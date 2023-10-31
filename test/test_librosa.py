"""
PoC using librosa + Support Vector Classifier
"""
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def main():
    # Define the path to the directory containing speaker audio samples
    data_dir = 'path/to/audio/samples'

    # Initialize empty lists to store features and labels
    features = []
    labels = []

    # Iterate over audio files in the data directory
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)

        # Load audio file using Librosa
        audio, sr = librosa.load(file_path, duration=3)  # Adjust duration as needed

        # Extract audio features (e.g., using Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(audio, sr=sr)
        features.append(np.mean(mfccs, axis=1))  # Use mean values as features

        # Extract label from the filename (assuming the filename contains the speaker's name)
        label = filename.split('_')[0]  # Adjust splitting logic based on your filename format
        labels.append(label)

    # Convert features and labels to numpy arrays
    X = np.array(features)
    y = np.array(labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a classifier (e.g., Support Vector Machine)
    clf = SVC()
    clf.fit(X_train, y_train)

    # Predict speaker labels for the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


if __name__ == '__main__':
    main()

