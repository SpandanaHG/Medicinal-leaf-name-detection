import os
import cv2
import numpy as np
from flask import Flask, request, render_template, url_for
from pymongo import MongoClient
from bson.binary import Binary
import pickle
import logging

# Flask app initialization
app = Flask(__name__)
UPLOAD_FOLDER = os.path.abspath(os.path.join('static', 'uploads'))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "medicinal_leaf_db"
COLLECTION_NAME = "leaves"

# Dataset Cache
DATASET_CACHE = "dataset_cache.pkl"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Global variables
dataset, labels, categories = None, None, None

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

def is_image_file(filename):
    """
    Checks if the file has an allowed image extension.
    """
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def convert_dataset_to_mongodb(dataset_dir):
    """
    Uploads the dataset to MongoDB, organizing images by their categories.
    """
    categories = os.listdir(dataset_dir)
    for category in categories:
        category_path = os.path.join(dataset_dir, category)
        if not os.path.isdir(category_path):
            continue

        # Example scientific names and usages (replace with actual data)
        scientific_name = "Scientific name for " + category  # Replace this with actual data
        usage = "Usage details for " + category  # Replace this with actual data

        for image_name in os.listdir(category_path):
            if not is_image_file(image_name):
                continue

            image_path = os.path.join(category_path, image_name)
            try:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                
                document = {
                    "category": category,
                    "image_name": image_name,
                    "image_data": Binary(image_data),
                    "scientific_name": scientific_name,  # Add scientific name
                    "usage": usage  # Add usage details
                }
                collection.insert_one(document)
                print(f"Uploaded {image_name} under category {category}")
            except Exception as e:
                print(f"Error uploading {image_name}: {e}")
    print("Dataset successfully uploaded to MongoDB.")


def load_dataset_from_mongodb():
    """
    Loads the dataset from MongoDB into memory for prediction.
    """
    global dataset, labels, categories
    dataset, labels = [], []
    categories = collection.distinct("category")
    
    if not categories:
        raise ValueError("No categories found in MongoDB.")
    
    print(f"Categories loaded from MongoDB: {categories}")  # Debugging statement
    
    for document in collection.find():
        image_binary = document["image_data"]
        category = document["category"]
        class_index = categories.index(category)

        try:
            # Decode the binary image and preprocess
            image = np.frombuffer(image_binary, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.resize(image, (224, 224))
                dataset.append(image)
                labels.append(class_index)
        except Exception as e:
            print(f"Error processing image: {e}")

    dataset = np.array(dataset)
    labels = np.array(labels)
    print(f"Loaded {len(dataset)} images from MongoDB.")  # Debugging statement

def load_dataset_from_cache():
    """
    Loads the dataset from a cached file if available.
    """
    global dataset, labels, categories
    if os.path.exists(DATASET_CACHE):
        try:
            with open(DATASET_CACHE, 'rb') as f:
                dataset, labels, categories = pickle.load(f)
                print(f"Dataset loaded from cache. Categories: {categories}")
                print(f"Dataset shape: {dataset.shape}, Labels shape: {labels.shape}")
                return True
        except Exception as e:
            print(f"Error loading dataset from cache: {e}")
    return False

def save_dataset_to_cache():
    """
    Saves the current dataset to a cache file for faster future loading.
    """
    try:
        with open(DATASET_CACHE, 'wb') as f:
            pickle.dump((dataset, labels, categories), f)
            print("Dataset cached to disk.")
    except Exception as e:
        print(f"Error saving dataset to cache: {e}")


def predict_leaf(image_path):
    """
    Predicts the leaf category based on the uploaded image and fetches details from MongoDB.
    """
    global dataset, labels, categories

    # Threshold for minimum similarity (adjust this value based on testing)
    SIMILARITY_THRESHOLD = 5000000  # Example value, adjust as needed

    # Validate that the dataset is loaded
    if dataset is None or labels is None or categories is None or dataset.size == 0 or labels.size == 0:
        print("Dataset not loaded or empty.")
        return "Dataset not loaded. Please check MongoDB.", None, None

    try:
        # Read and preprocess the input image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Error: Unable to read input image at {image_path}")
            return "Error: Unable to read input image.", None, None

        image = cv2.resize(image, (224, 224))

        # Calculate differences and find the closest match
        differences = [
            (np.sum(np.abs(image - data_image)), labels[i]) for i, data_image in enumerate(dataset)
        ]
        closest_difference, closest_label = min(differences, key=lambda x: x[0])

        # Check if the closest match is within the similarity threshold
        if closest_difference > SIMILARITY_THRESHOLD:
            return "No match found", None, None

        leaf_name = categories[closest_label]

        # Fetch additional details (scientific name and usage) from MongoDB
        document = collection.find_one({"category": leaf_name})
        if document:
            scientific_name = document.get("scientific_name", "N/A")
            usage = document.get("usage", "N/A")
            return leaf_name, scientific_name, usage
        else:
            return leaf_name, "N/A", "N/A"
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return "Error in processing image.", None, None


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles the Flask app routes and processes uploaded images.
    """
    leaf_name = None
    scientific_name = None
    usage = None
    uploaded_image_url = None

    if request.method == 'POST':
        file = request.files['file']
        if file and is_image_file(file.filename):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            try:
                file.save(file_path)

                # Predict the leaf name and get details from MongoDB
                leaf_name, scientific_name, usage = predict_leaf(file_path)

                # Display the uploaded image and result
                uploaded_image_url = url_for('static', filename=f'uploads/{file.filename}')
            except Exception as e:
                print(f"Error processing the uploaded file: {e}")
                return render_template("index.html", uploaded_image=None, result="Error in processing image.")

    return render_template(
        "index.html",
        uploaded_image=uploaded_image_url,
        leaf_name=leaf_name,
        scientific_name=scientific_name,
        usage=usage
    )

if __name__ == '__main__':
    print("Starting Flask app...")
    
    try:
        # Check if dataset is already loaded into MongoDB
        if collection.count_documents({}) == 0:  # Check if MongoDB is empty
            print("No data found in MongoDB. Uploading dataset...")
            # Load dataset to MongoDB only once
            convert_dataset_to_mongodb(r"C:\medicinal_leaf_detection\Medicinal Leaf dataset")
        else:
            print("Dataset already loaded in MongoDB.")
        
        # Load dataset from MongoDB to memory
        load_dataset_from_mongodb()
        save_dataset_to_cache()  # Cache the dataset for future use

    except Exception as e:
        print(f"Error: {e}")

    # Disable debug and reloader
    app.run(debug=True, port=5000, use_reloader=False)     