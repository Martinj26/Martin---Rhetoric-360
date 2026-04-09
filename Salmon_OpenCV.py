import cv2
import numpy as np

# === INPUT IMAGES ===
main_image_path = "Still Image of Salmon.tif"   # big image with many salmon
template_path   = "Week 10 Sample.tif"       # cropped image of one salmon
threshold = 0.2
# detection threshold (0–1). Raise to reduce false detections.

# === LOAD IMAGES ===
img = cv2.imread(main_image_path)
template = cv2.imread(template_path)

if img is None:
    raise FileNotFoundError("Main image not found. Check filename and folder.")
if template is None:
    raise FileNotFoundError("Template image not found. Check filename and folder.")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

w, h = template_gray.shape[::-1]

# === TEMPLATE MATCHING (CROSS CORRELATION) ===
result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

# === FIND MATCH LOCATIONS ABOVE THRESHOLD ===
locations = np.where(result >= threshold)
points = list(zip(*locations[::-1]))  # convert to (x,y)

# === OPTIONAL: SUPPRESS NEAR-DUPLICATES ===
filtered_points = []
min_dist = min(w, h) * 0.6  # distance to consider points separate

for pt in points:
    if all(np.linalg.norm(np.array(pt) - np.array(fp)) > min_dist for fp in filtered_points):
        filtered_points.append(pt)

# === DRAW DETECTIONS ===
for pt in filtered_points:
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

# === PRINT COUNT ===
print(f"Detected Salmon: {len(filtered_points)}")

# === SHOW OUTPUT IMAGE ===
cv2.imshow("Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
