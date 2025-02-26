import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import imutils


image_path = "Images/images.png"  # Replace with your image path
image = cv2.imread(image_path)

# **Preprocessing Steps**
# Convert to Grayscale (Removes color noise)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binarization (Thresholding - Increases contrast)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Noise Removal (Removes unwanted dots)
denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)

# Edge Detection (Helps find text regions)
edges = cv2.Canny(denoised, 100, 200)

# Deskewing (Straightens tilted text)
coords = np.column_stack(np.where(edges > 0))
angle = cv2.minAreaRect(coords)[-1]

# Adjust angle range
if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle

deskewed = imutils.rotate_bound(image, angle)

# **Initialize EasyOCR reader**
reader = easyocr.Reader(['en'])

# **Perform OCR on the processed image**
results = reader.readtext(deskewed)

# **Extract detected text**
detected_text = " ".join([text for _, text, _ in results])
print("Extracted Text:", detected_text)

# **Store extracted text in a file**
text_file_path = "extracted_text/extracted_text.txt"
with open(text_file_path, "w", encoding="utf-8") as file:
    file.write(detected_text)

print(f"Extracted text saved to {text_file_path}")

# **Draw bounding boxes on the image**
for (bbox, text, confidence) in results:
    top_left = tuple(map(int, bbox[0]))
    bottom_right = tuple(map(int, bbox[2]))
    cv2.rectangle(deskewed, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(deskewed, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# Convert image color for displaying
display_image = cv2.cvtColor(deskewed, cv2.COLOR_BGR2RGB)

# **Show the result**
plt.imshow(display_image)
plt.axis("off")
plt.show()
