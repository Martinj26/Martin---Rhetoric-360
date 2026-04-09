import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Load image ---
img = cv2.imread("Sample 1.CR2")   # change this
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

h, w = gray.shape

# --- Function to sample a line ---
def sample_line(img, x0, y0, theta, length=200):
    values = []
    for t in range(-length//2, length//2):
        x = int(x0 + t * np.cos(theta))
        y = int(y0 + t * np.sin(theta))
        
        if 0 <= x < w and 0 <= y < h:
            values.append(img[y, x])
    
    return np.array(values)

# --- Sweep angles ---
angles = np.arange(0, 180, 5)
contrast_values = []

# sample around center of image
cx, cy = w // 2, h // 2

for angle in angles:
    theta = np.deg2rad(angle)
    
    variances = []
    
    # sample multiple parallel lines (shift perpendicular)
    for offset in range(-50, 50, 10):
        x_shift = int(cx + offset * np.cos(theta + np.pi/2))
        y_shift = int(cy + offset * np.sin(theta + np.pi/2))
        
        line = sample_line(gray, x_shift, y_shift, theta)
        
        if len(line) > 10:
            variances.append(np.var(line))
    
    # average variance for this angle
    if len(variances) > 0:
        contrast_values.append(np.mean(variances))
    else:
        contrast_values.append(0)

# --- Plot result ---
plt.plot(angles, contrast_values)
plt.xlabel("Angle (degrees)")
plt.ylabel("Contrast (variance)")
plt.title("Angle vs Texture Contrast")
plt.show()