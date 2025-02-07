import os
import json
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response
from werkzeug.utils import secure_filename
from PIL import Image, ImageTk
import io
import random
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load metadata
def load_json_metadata(json_path):
    with open(json_path, 'r') as file:
        metadata = json.load(file)
    return metadata

metadata = load_json_metadata('metadata.json')

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_and_compute(image, detector):
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(des1, des2, matcher):
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def resize_image(image, max_width=800, max_height=600):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    if width > max_width or height > max_height:
        if aspect_ratio > 1:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized_image = image
    return resized_image

def draw_matches(img1, kp1, img2, kp2, matches):
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        matchColor=(0, 255, 0),  # Green color for matches
        singlePointColor=(255, 0, 0),  # Red color for keypoints
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    overlay = img_matches.copy()
    for match in matches:
        pt1 = tuple(np.round(kp1[match.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(kp2[match.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random color
        cv2.circle(overlay, pt1, 5, (0, 0, 255), 5)  # Red circles for keypoints in img1
        cv2.circle(overlay, pt2, 5, (0, 0, 255), 5)  # Red circles for keypoints in img2
        cv2.line(overlay, pt1, pt2, color, 5)  # Random color lines for matches
    cv2.addWeighted(overlay, 0.9, img_matches, 0.9, 0, img_matches)  # Apply transparency
    return img_matches

def resize_to_match(image1, image2, keypoints1, keypoints2):
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    if width1 > width2 or height1 > height2:
        scale_x = width2 / width1
        scale_y = height2 / height1
        resized_image1 = cv2.resize(image1, (width2, height2), interpolation=cv2.INTER_AREA)
        resized_keypoints1 = [cv2.KeyPoint(p.pt[0] * scale_x, p.pt[1] * scale_y, p.size) for p in keypoints1]
        return resized_image1, image2, resized_keypoints1, keypoints2
    else:
        scale_x = width1 / width2
        scale_y = height1 / height2
        resized_image2 = cv2.resize(image2, (width1, height1), interpolation=cv2.INTER_AREA)
        resized_keypoints2 = [cv2.KeyPoint(p.pt[0] * scale_x, p.pt[1] * scale_y, p.size) for p in keypoints2]
        return image1, resized_image2, keypoints1, resized_keypoints2

camera = cv2.VideoCapture(0)
output_frame = None
lock = threading.Lock()
matches_info = []

def generate_frames():
    global output_frame, lock, matches_info
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    while True:
        success, frame = camera.read()
        if not success:
            break
        with lock:
            output_frame = frame.copy()
        
        query_image = frame.copy()
        query_keypoints, query_descriptors = detect_and_compute(query_image, orb)

        images = []
        filenames = []
        for filename in os.listdir('Collection'):
            img = cv2.imread(os.path.join('Collection', filename), cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
                filenames.append(filename)

        matches_info = []
        for img, filename in zip(images, filenames):
            img_keypoints, img_descriptors = detect_and_compute(img, orb)
            matches = match_features(query_descriptors, img_descriptors, bf)

            MIN_MATCH_COUNT = 10
            if len(matches) > MIN_MATCH_COUNT:
                query_image_resized, img_resized, query_keypoints_resized, img_keypoints_resized = resize_to_match(query_image, img, query_keypoints, img_keypoints)
                img_matches = draw_matches(query_image_resized, query_keypoints_resized, img_resized, img_keypoints_resized, matches[:10])
                _, img_encoded = cv2.imencode('.jpg', img_matches)
                img_b64 = base64.b64encode(img_encoded).decode('utf-8')

                info = metadata.get(filename, {})
                matches_info.append({
                    'image': img_b64,
                    'filename': filename,
                    'description': info.get('description', 'N/A'),
                    'author': info.get('author', 'N/A'),
                    'publication_year': info.get('publication_year', 'N/A'),
                    'genre': info.get('genre', 'N/A')
                })
                logging.info(f"Match found: {filename} - {info.get('description', 'N/A')}")
            else:
                matches_info.append({
                    'image': None,
                    'filename': filename,
                    'description': 'Not enough matches found',
                    'author': '',
                    'publication_year': '',
                    'genre': ''
                })

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_camera')
def live_camera():
    return render_template('live_camera.html', matches_info=matches_info)

@app.route('/capture_image', methods=['POST'])
def capture_image():
    global output_frame, lock
    with lock:
        if output_frame is None:
            return 'No frame captured', 400
        filename = f"capture_{int(time.time())}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, output_frame)
    return redirect(url_for('result', filenames=filename))

@app.route('/live_match')
def live_match():
    global output_frame, lock
    with lock:
        if output_frame is None:
            return 'No frame captured', 400
        query_image = output_frame.copy()
    
    orb = cv2.ORB_create()
    query_keypoints, query_descriptors = detect_and_compute(query_image, orb)

    images = []
    filenames = []
    for filename in os.listdir('Collection'):
        img = cv2.imread(os.path.join('Collection', filename), cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
            filenames.append(filename)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_info = []

    for img, filename in zip(images, filenames):
        img_keypoints, img_descriptors = detect_and_compute(img, orb)
        matches = match_features(query_descriptors, img_descriptors, bf)

        MIN_MATCH_COUNT = 10
        if len(matches) > MIN_MATCH_COUNT:
            query_image_resized, img_resized, query_keypoints_resized, img_keypoints_resized = resize_to_match(query_image, img, query_keypoints, img_keypoints)
            img_matches = draw_matches(query_image_resized, query_keypoints_resized, img_resized, img_keypoints_resized, matches[:10])
            _, img_encoded = cv2.imencode('.jpg', img_matches)
            img_b64 = base64.b64encode(img_encoded).decode('utf-8')

            info = metadata.get(filename, {})
            matches_info.append({
                'image': img_b64,
                'filename': filename,
                'description': info.get('description', 'N/A'),
                'author': info.get('author', 'N/A'),
                'publication_year': info.get('publication_year', 'N/A'),
                'genre': info.get('genre', 'N/A')
            })
        else:
            matches_info.append({
                'image': None,
                'filename': filename,
                'description': 'Not enough matches found',
                'author': '',
                'publication_year': '',
                'genre': ''
            })

    return render_template('result.html', matches_info=matches_info)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'files' not in request.files:
            return 'No file part'
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return 'No selected files'
        filenames = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                filenames.append(filename)
        return redirect(url_for('result', filenames=','.join(filenames)))
    return render_template('index.html')

@app.route('/result/<filenames>')
def result(filenames):
    filenames = filenames.split(',')
    matches_info = []

    for filename in filenames:
        query_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        query_image = cv2.imread(query_image_path, cv2.IMREAD_COLOR)
        if query_image is None:
            continue

        orb = cv2.ORB_create()
        query_keypoints, query_descriptors = detect_and_compute(query_image, orb)

        images = []
        collection_filenames = []
        for collection_filename in os.listdir('Collection'):
            img = cv2.imread(os.path.join('Collection', collection_filename), cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
                collection_filenames.append(collection_filename)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        for img, collection_filename in zip(images, collection_filenames):
            img_keypoints, img_descriptors = detect_and_compute(img, orb)
            matches = match_features(query_descriptors, img_descriptors, bf)

            MIN_MATCH_COUNT = 10
            if len(matches) > MIN_MATCH_COUNT:
                query_image_resized, img_resized, query_keypoints_resized, img_keypoints_resized = resize_to_match(query_image, img, query_keypoints, img_keypoints)
                img_matches = draw_matches(query_image_resized, query_keypoints_resized, img_resized, img_keypoints_resized, matches[:10])
                _, img_encoded = cv2.imencode('.jpg', img_matches)
                img_b64 = base64.b64encode(img_encoded).decode('utf-8')

                info = metadata.get(collection_filename, {})
                matches_info.append({
                    'image': img_b64,
                    'filename': collection_filename,
                    'description': info.get('description', 'N/A'),
                    'author': info.get('author', 'N/A'),
                    'publication_year': info.get('publication_year', 'N/A'),
                    'genre': info.get('genre', 'N/A')
                })
            else:
                matches_info.append({
                    'image': None,
                    'filename': collection_filename,
                    'description': 'Not enough matches found',
                    'author': '',
                    'publication_year': '',
                    'genre': ''
                })

        os.remove(query_image_path)  # Delete the uploaded file after processing

    return render_template('result.html', matches_info=matches_info)

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=5000, debug=True)

