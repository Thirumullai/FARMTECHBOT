import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, jsonify
import os
import os.path
from PIL import Image
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pandas as pd
from flask import Flask, render_template, redirect, url_for, request, jsonify
import imagehash
import base64
app = Flask(__name__)

# Assuming you have a dataset directory containing images of pests
dataset_dir = "pestdatasets"

# Load dataset for pest detection
X = []  # Feature vectors
y = []  # Labels
image_names = [] # Store image names for matching
image_hashes = [] # Store image hashes for matching

# Loop through all subdirectories (pest classes) in the dataset directory
for pest_class in os.listdir(dataset_dir):
    pest_class_dir = os.path.join(dataset_dir, pest_class)
    if os.path.isdir(pest_class_dir):
        # Loop through all image files in the current pest class directory
        for img_file in os.listdir(pest_class_dir):
            img_path = os.path.join(pest_class_dir, img_file)
            if os.path.isfile(img_path):
                img = imread(img_path, as_gray=True)  # Read image in grayscale
                img = resize(img, (100, 100))  # Resize image to a common size
                # Extract HOG features
                features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
                X.append(features)
                y.append(pest_class)
                image_names.append(img_file)
                image_hashes.append(str(imagehash.phash(Image.open(img_path))))
# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Train a classifier for pest detection
pest_detection_clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0, probability=True))
pest_detection_clf.fit(X, y)

# Function to predict pest given an image path
def predict_pest(image_path):
    img = imread(image_path, as_gray=True)
    img = resize(img, (100, 100))
    features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    predicted_label = pest_detection_clf.predict([features])[0]
    return predicted_label

# Function to check if the uploaded image is present in the dataset
def check_exact_match(image_path):
    uploaded_hash = str(imagehash.phash(Image.open(image_path)))
    for i, dataset_hash in enumerate(image_hashes):
        if uploaded_hash == dataset_hash:
            return y[i] # Return label if an exact match is found
    return None # Return None if no exact match is found

# Download NLTK resources for chatbot
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text using NLTK for chatbot
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    words = [word for word in tokens if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    preprocessed_text = ' '.join(words)
    return preprocessed_text

# Load datasets and preprocess for chatbot
soil_testing_data = pd.read_csv('soil.csv')
pest_detection_data = pd.read_csv('pest.csv')
responses_data = pd.read_csv('response.csv')
combined_data = pd.concat([soil_testing_data, pest_detection_data], ignore_index=True)
combined_data.dropna(inplace=True)
combined_data['Preprocessed Question'] = combined_data['Question'].apply(preprocess_text)
combined_data = combined_data[combined_data['Preprocessed Question'].map(len) > 0]

# Train machine learning model for chatbot
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(combined_data['Preprocessed Question'])
y_train = combined_data['Intent']
chatbot_clf = LinearSVC()
chatbot_clf.fit(X_train, y_train)

# Function to classify intent for chatbot
def classify_intent(input_text):
    preprocessed_input = preprocess_text(input_text)
    if not preprocessed_input:
        return None
    input_vector = vectorizer.transform([preprocessed_input])
    intent = chatbot_clf.predict(input_vector)
    return intent[0] if len(intent) > 0 else None

# Function to retrieve a single response per intent for chatbot
def get_response(intent):
    if intent is None:
        return "I'm sorry, I couldn't understand your query. Please try again."
    responses = responses_data.loc[responses_data['Intent'] == intent, 'Response'].tolist()
    if responses:
        return responses[0]
    else:
        return "I'm sorry, I don't have a response for that query."
# Function to store phone numbers in a CSV file
def store_phone_number(phone_number):
    data = {'Phone Numbers': [phone_number]}
    df = pd.DataFrame(data)
    if not os.path.exists('phone_numbers.csv'):
        df.to_csv('phone_numbers.csv', index=False)
    else:
        existing_df = pd.read_csv('phone_numbers.csv')
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_csv('phone_numbers.csv', index=False)
# Assuming you have other routes defined already

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Check login credentials if needed
        # For simplicity, let's assume there are no credentials needed
        # If login is successful, redirect to chatbot index page
        return redirect(url_for('chatbot_index'))
    return render_template('login.html')

@app.route('/chatbot')
def chatbot_index():
    return render_template('index.html')
@app.route('/store_phone', methods=['POST'])
def store_phone():
    phone_number = request.form['phone_number']
    store_phone_number(phone_number)
    return redirect(url_for('chatbot_index'))

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    img_bytes = file.read()
    img = Image.open(BytesIO(img_bytes))

    # Save the image to a temporary file
    temp_image_path = 'temp_image.jpg'
    img.save(temp_image_path)

    # Check if the uploaded image is an exact match with any image in the dataset
    matched_label = check_exact_match(temp_image_path)

    # Delete the temporary file
    os.remove(temp_image_path)

    if matched_label is not None:
        return jsonify({'pest_type': matched_label})
    else:
        return jsonify({'message': 'No match found for the uploaded image'})
   

@app.route('/process_input', methods=['POST'])
def process_input():
    user_input = request.json['input']
    predicted_intent = classify_intent(user_input)
    response = get_response(predicted_intent)
    return jsonify({'response': response})


#Route to switch from English to Tamil
@app.route('/switch-to-tamil')
def switch_to_tamil():
    return redirect("http://localhost:5001/") 

if __name__ == '__main__':
    app.run(debug=True)
