from PIL import Image
import math

# ---------------- Activation functions ----------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tanh(x):
    return math.tanh(x)

# ---------------- Neural network weights ----------------
# Inputs: [elongation, pixel_density]

W1 = [
    [3.2, 1.5],
    [2.1, 2.7]
]
b1 = [-2.5, -2.0]

W2 = [2.2, 2.8]
b2 = -1.8

def forward_pass(x):
    h = []
    for i in range(2):
        z = W1[i][0]*x[0] + W1[i][1]*x[1] + b1[i]
        h.append(tanh(z))

    z_out = W2[0]*h[0] + W2[1]*h[1] + b2
    return sigmoid(z_out)

# ---------------- Feature extraction ----------------

def extract_features(window):
    width, height = window.size
    pixels = list(window.getdata())

    threshold = 120
    active = [(i % width, i // width) for i, p in enumerate(pixels) if p < threshold]

    if len(active) == 0:
        return [0.0, 0.0]

    xs = [p[0] for p in active]
    ys = [p[1] for p in active]

    elongation = (max(xs) - min(xs) + 1) / (max(ys) - min(ys) + 1)
    density = len(active) / (width * height)

    return [elongation, density]

# ---------------- Sliding window salmon count ----------------

def count_salmon(image_path):
    img = Image.open(image_path).convert("L")

    window_size = 64
    stride = 32
    salmon_count = 0

    for y in range(0, img.height - window_size, stride):
        for x in range(0, img.width - window_size, stride):
            window = img.crop((x, y, x + window_size, y + window_size))
            features = extract_features(window)

            prob = forward_pass(features)

            if prob > 0.9019:
                salmon_count += 1



    return salmon_count

# ---------------- Run ----------------

count = count_salmon("Salmon.jpg")
print("Estimated salmon count:", count)
