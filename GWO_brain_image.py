import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Read MRI image
# -----------------------------
image = cv2.imread("brain_mri.jpg", cv2.IMREAD_GRAYSCALE)

# -----------------------------
# Step 2: Define Fitness Function
# Using Otsu-based between-class variance
# -----------------------------
def fitness(thresholds, img):
    thresholds = np.sort(thresholds)
    bins = [0] + list(thresholds) + [256]
    hist, _ = np.histogram(img, bins=256, range=(0,256))
    hist = hist / img.size
    total_fitness = 0
    for i in range(len(bins)-1):
        p = hist[bins[i]:bins[i+1]]
        if len(p) > 0:
            total_fitness += np.var(p) * np.sum(p)
    return total_fitness

# -----------------------------
# Step 3: Initialize GWO
# -----------------------------
num_wolves = 20
num_thresholds = 3   # For 4 regions
max_iter = 50

# Wolves positions (threshold candidates)
positions = np.random.randint(0, 256, size=(num_wolves, num_thresholds))

# Alpha, Beta, Delta initialization
Alpha_pos = np.zeros(num_thresholds)
Alpha_score = -np.inf
Beta_pos = np.zeros(num_thresholds)
Beta_score = -np.inf
Delta_pos = np.zeros(num_thresholds)
Delta_score = -np.inf

# -----------------------------
# Step 4: GWO Algorithm
# -----------------------------
for t in range(max_iter):
    a = 2 - t * (2 / max_iter)  # linearly decreasing
    
    for i in range(num_wolves):
        # Clip thresholds to valid range
        positions[i] = np.clip(positions[i], 0, 255)
        fit = fitness(positions[i], image)
        
        # Update Alpha, Beta, Delta
        if fit > Alpha_score:
            Alpha_score = fit
            Alpha_pos = positions[i].copy()
        elif fit > Beta_score:
            Beta_score = fit
            Beta_pos = positions[i].copy()
        elif fit > Delta_score:
            Delta_score = fit
            Delta_pos = positions[i].copy()
    
    # Update positions of wolves
    for i in range(num_wolves):
        for j in range(num_thresholds):
            r1 = np.random.rand()
            r2 = np.random.rand()
            
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * Alpha_pos[j] - positions[i][j])
            X1 = Alpha_pos[j] - A1 * D_alpha
            
            r1 = np.random.rand()
            r2 = np.random.rand()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * Beta_pos[j] - positions[i][j])
            X2 = Beta_pos[j] - A2 * D_beta
            
            r1 = np.random.rand()
            r2 = np.random.rand()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * Delta_pos[j] - positions[i][j])
            X3 = Delta_pos[j] - A3 * D_delta
            
            # Update wolf position
            positions[i][j] = (X1 + X2 + X3) / 3

# -----------------------------
# Step 5: Segment Image
# -----------------------------
best_thresholds = np.sort(Alpha_pos)
segmented = np.zeros_like(image)

segmented[(image >= 0) & (image < best_thresholds[0])] = 0
segmented[(image >= best_thresholds[0]) & (image < best_thresholds[1])] = 85
segmented[(image >= best_thresholds[1]) & (image < best_thresholds[2])] = 170
segmented[(image >= best_thresholds[2])] = 255

# -----------------------------
# Step 6: Visualize
# -----------------------------
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title("Original MRI")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(segmented, cmap='gray')
plt.title("Segmented MRI (GWO)")
plt.axis('off')
plt.show()
