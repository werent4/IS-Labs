import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier  # Import MLPClassifier
from sklearn.metrics import accuracy_score  # For evaluating the model


def load_and_preprocess_image(image_path, display=False):
    """Loads an image, performs binarization, and optionally displays it."""
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to load image at path: {image_path}")

    # Binarize the image
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Optionally display the image
    if display:
        cv2.imshow("Binarized Image", binary_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return binary_image


def segment_digits(binary_image, h_=20):
    """Segments digits from the image and returns them as a list of images."""
    # Find contours on the binarized image
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort contours from left to right
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    # Digit segmentation with size filtering
    digits = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > h_:  # Filter contours based on minimum and maximum sizes
            digit = binary_image[y : y + h, x : x + w]
            resized_digit = cv2.resize(digit, (28, 28))  # Resize to 28x28
            digits.append(resized_digit)

    return digits


def extract_features(images):
    """Extract features from a list of images by flattening them into vectors."""
    return np.array([image.flatten() / 255.0 for image in images])  # Normalize pixels


def visualize_digits(digits, predictions=None):
    """Displays segmented digits, enlarged for better viewing."""
    for i, digit in enumerate(digits):
        # Enlarge the digit image for display
        enlarged_digit = cv2.resize(digit, (200, 200), interpolation=cv2.INTER_LINEAR)

        # Window name with predicted digit (if available)
        window_name = f"Digit {i}"
        if predictions is not None:
            window_name += f": Predicted {predictions[i]}"

        # Create a larger window and display the image
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 200, 200)
        cv2.imshow(window_name, enlarged_digit)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        # Path to the image
        train_image_path = "digits.jpg"

        # Load and preprocess the training image
        binary_train_image = load_and_preprocess_image(train_image_path)

        # Segment digits from the training image
        train_digits = segment_digits(binary_train_image)
        num_train_digits = len(train_digits)
        print(f"Number of segmented digits from training image: {num_train_digits}")

        # Check if the number of segmented digits is correct for training
        if num_train_digits != 10:
            print(
                f"Error: found {num_train_digits} digits, but expected 10 for training."
            )
            visualize_digits(train_digits)
            exit(1)

        # Extract features from training digits
        train_features = extract_features(train_digits)

        # Labels for the training digits
        train_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Initialize and train the MLP classifier
        mlp = MLPClassifier(
            hidden_layer_sizes=(50,),  # One hidden layer with 50 neurons
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        mlp.fit(train_features, train_labels)

        # Path to the test image
        test_image_path = "test1.jpg"

        # Load and preprocess the test image
        binary_test_image = load_and_preprocess_image(test_image_path)

        # Segment digits from the test image
        test_digits = segment_digits(binary_test_image)
        num_test_digits = len(test_digits)
        print(f"Number of segmented digits from test image: {num_test_digits}")

        # Check if there are any segmented digits
        if num_test_digits == 0:
            print("Error: No digits were found in the test image.")
            visualize_digits(test_digits)
            exit(1)

        # Extract features from test digits
        test_features = extract_features(test_digits)

        # Predictions on the test set
        test_predictions = mlp.predict(test_features)

        # Print predictions for each digit
        print(f"Predictions for test digits: {test_predictions}")

        # Visualize segmented and predicted test digits
        visualize_digits(test_digits, test_predictions)

    except FileNotFoundError as e:
        print(e)
