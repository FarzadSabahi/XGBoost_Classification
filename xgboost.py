import xgboost as xgb
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Preprocess the data by flattening the images
# Each 32x32 image is flattened into a 3072-feature vector (32*32*3, because it's a color image with 3 channels)
X_train = train_images.reshape(train_images.shape[0], -1)
X_test = test_images.reshape(test_images.shape[0], -1)

# Convert labels from 2D to 1D array (required for XGBoost)
y_train = train_labels.flatten()
y_test = test_labels.flatten()

# Create an XGBoost classifier instance
# Parameters are set to basic values but can be tuned for better performance
clf = xgb.XGBClassifier(
    objective='multi:softprob',  # softmax probability for multiclass classification
    num_class=10,  # number of classes
    eval_metric='mlogloss',  # loss function
    seed=42,  # for reproducibility
    use_label_encoder=False  # to avoid a warning related to future updates
)

# Train the classifier
clf.fit(X_train, y_train)

# Predict the test set results
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100.0:.2f}%")
