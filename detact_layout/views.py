import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage 
import pytesseract
import cv2
from PIL import Image
import numpy as np
import re

# Set Tesseract command
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def clean_text(text):
    # Example of cleaning common OCR errors
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    # Fine-tuning regex to remove unwanted characters, but keep necessary punctuation
    text = re.sub(r'[^\w\s,.!?;:()"\']', '', text)  # Keep common punctuation
    return text.strip()

def home(request):
    return render(request, 'layout/index.html')

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Histogram equalization to enhance contrast
    gray = cv2.equalizeHist(gray)
    
    # Increase contrast further if necessary
    gray = cv2.convertScaleAbs(gray, alpha=2.5, beta=50)
    
    # Resize the image to increase readability (aggressively)
    height, width = gray.shape
    scale_factor = 5  # Increase the scale factor for more aggressive resizing
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized_image = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply Gaussian Blur to remove noise
    blurred_image = cv2.GaussianBlur(resized_image, (3, 3), 0)
    
    # Apply adaptive thresholding and combine with Otsu's method
    _, binary_otsu = cv2.threshold(blurred_image, 0, 355, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_adaptive = cv2.adaptiveThreshold(blurred_image, 355, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Combine the results
    binary = cv2.bitwise_and(binary_adaptive, binary_otsu)
    
    # Sharpen the image
    kernel_sharpening = np.array([[0, -1, 0], 
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    binary = cv2.filter2D(binary, -1, kernel_sharpening)
    
    # Apply dilation and erosion to remove noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=5)
    binary = cv2.erode(binary, kernel, iterations=5)
    
    return binary

def upload(request):
    if request.method == 'POST' and request.FILES.get('document'):
        document = request.FILES['document']
        fs = FileSystemStorage()
        filename = fs.save(document.name, document)
        uploaded_file_url = fs.url(filename)

        # Preprocess the uploaded document
        try:
            preprocessed_image = preprocess_image(fs.path(filename))
        except ValueError as e:
            return render(request, 'layout/index.html', {'error': str(e)})

        # Convert the processed image to PIL format
        preprocessed_image_pil = Image.fromarray(preprocessed_image)

        # Extract text using Tesseract
        custom_config = r'--oem 3 --psm 6 --dpi 300'
        text = pytesseract.image_to_string(preprocessed_image_pil, config=custom_config)

        # Clean the extracted text
        text = clean_text(text)

        context = {
            'uploaded_file_url': uploaded_file_url,
            'text': text,
        }
        return render(request, 'layout/index.html', context)
    return render(request, 'layout/index.html')
