import os
import cv2
import joblib
import numpy as np
from sklearn import svm
from utils import extract_hand_landmarks

# DATA_DIR = r"data\dataset"
# MODEL_PATH = "svm_model.pkl"
DATA_DIR = r"data/data_num"
MODEL_PATH = "svm_model2.pkl"

X_train, y_train = [], []

# Loop through dataset folders
for label in os.listdir(DATA_DIR):
    folder = os.path.join(DATA_DIR, label)
    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        if img is None:
            continue
        # print(f"file is {file} image is {img_path}")
        features = extract_hand_landmarks(img)
        if features is not None:
            print(f"file is {file} image is {img_path} label is {label}")
            X_train.append(features)
            y_train.append(label)

print(f"Collected {len(X_train)} samples.")

# Train SVM classifier
# clf = svm.SVC(kernel="linear", probability=True)
# clf.fit(X_train, y_train)

clf = svm.SVC(kernel="poly", degree=3, gamma="scale", coef0=1, probability=True)
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# X_train -> features- need to reduce
# y_train -> labels - no need to reduce
'''
#2D plotting
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Convert lists to numpy arrays
X = np.array(X_train)
y = np.array(y_train)

# Reduce to 2D with PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

#train new svm on pca values
clf_pca = svm.SVC(kernel="linear")
clf_pca.fit(X_reduced, y)

# Plot clusters
plt.figure(figsize=(8,6))
for label in np.unique(y):
    idx = y == label
    plt.scatter(X_reduced[idx, 0], X_reduced[idx, 1], label=label)

# Plot hyperplane
w = clf_pca.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(X_reduced[:,0].min()-1, X_reduced[:,0].max()+1, 100)
yy = a * xx - (clf_pca.intercept_[0]) / w[1]

plt.plot(xx, yy, "k-", linewidth=2)
plt.legend()
plt.title("Clusters of Hand Gestures (PCA reduced)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
'''

#3d plotting

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Convert lists to numpy arrays
X = np.array(X_train)
y = np.array(y_train)

# Reduce to 3D with PCA
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)

#train new svm on pca reduced values
clf_pca = svm.SVC(kernel="poly", degree=3, gamma="scale", coef0=1)
clf_pca.fit(X_reduced, y)

# Plot clusters in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

for label in np.unique(y):
    idx = y == label
    ax.scatter(X_reduced[idx, 0], X_reduced[idx, 1], X_reduced[idx, 2], label=label)


# Approximate decision boundary:
# Sample a 3D grid of points (coarse to keep it fast)
xx, yy, zz = np.meshgrid(
    np.linspace(X_reduced[:,0].min()-1, X_reduced[:,0].max()+1, 30),
    np.linspace(X_reduced[:,1].min()-1, X_reduced[:,1].max()+1, 30),
    np.linspace(X_reduced[:,2].min()-1, X_reduced[:,2].max()+1, 30)
)

grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
preds = clf_pca.predict(grid_points)

# Plot only boundary points (where classes change)
# For simplicity, show one class region as semi-transparent
mask = preds == np.unique(y)[0]
ax.scatter(grid_points[mask,0], grid_points[mask,1], grid_points[mask,2],
           alpha=0.05, color="yellow")

ax.set_title("3D PCA Clusters of Hand Gestures")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()
plt.show()
