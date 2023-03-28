"""
Extract features using convolutional filters,
    then take the features and use for image
    classification using Random Forest
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

from keras.models import Model, Sequential
from keras.layers import (
    Dense, Flatten, Conv2D, MaxPooling2D)
from keras.layers import BatchNormalization

from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model

import os
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


"""
Deep learning parts are memory intensive, so its better
        to use a smaller image dimension size
"""
size = 128


"""
Prepare training images and their labels
"""
train_images = []
train_labels = []

path_images_train = "IMAGES/PCAReconstructed/train/*"

for directory_path in glob.glob(path_images_train):
    label = directory_path.split("\\")[-1]
    # print(label)

    for image_path in glob.glob(
            os.path.join(directory_path, "*.*")):
        # print(image_path)

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (size, size))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        train_images.append(img)
        train_labels.append(label)


train_images = np.array(train_images)
train_labels = np.array(train_labels)


"""
Prepare testing images and their labels
"""
test_images = []
test_labels = []

path_images_test = "IMAGES/PCAReconstructed/validation/*"

for directory_pat in glob.glob(path_images_test):
    labels = directory_pat.split("\\")[-1]
    # print(labels)

    for image_pat in glob.glob(
            os.path.join(directory_pat, "*.*")):
        # print(image_pat)

        imag = cv2.imread(image_pat, cv2.IMREAD_COLOR)
        imag = cv2.resize(imag, (size, size))
        imag = cv2.cvtColor(imag, cv2.COLOR_RGB2BGR)

        test_images.append(imag)
        test_labels.append(labels)


test_images = np.array(test_images)
test_labels = np.array(test_labels)


"""
Encode labels from text to integers
"""
label_encoder = preprocessing.LabelEncoder()

label_encoder.fit(test_labels)
test_labels_encoded = label_encoder.transform(test_labels)

label_encoder.fit(train_labels)
train_labels_encoded = label_encoder.transform(train_labels)


"""
Split data into train and test datasets.
    - Not actually splitting, but just performing
        name reassignment. 
"""
X_train, y_train, X_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded


"""
Scale pixel values between 0 and 1
"""
X_train, X_test = X_train/255.0, X_test/255.0


"""
One-hot encode y values for neural network
"""
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)


"""
CONVOLUTIONAL FEATURE EXTRACTION 
CONVOLUTIONAL FEATURE EXTRACTION 
CONVOLUTIONAL FEATURE EXTRACTION 
"""

activation = 'sigmoid'

feature_extractor = Sequential()
feature_extractor.add(Conv2D(
    32, 3,
    activation=activation,
    padding='same',
    input_shape=(size, size, 3)))

feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(
    32, 3,
    activation=activation,
    padding='same',
    kernel_initializer='he_uniform'))

feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Conv2D(
    64, 3,
    activation=activation,
    padding='same',
    kernel_initializer='he_uniform'))

feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(
    64, 3, activation=activation,
    padding='same',
    kernel_initializer='he_uniform'))

feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Flatten())


"""
Add layers for deep learning prediction
"""
x = feature_extractor.output
x = Dense(
    128,
    activation=activation,
    kernel_initializer='he_uniform')(x)

prediction_layer = Dense(
    2,
    # should use sigmoid for binary classification
    activation='sigmoid')(x)


"""
Make a new model combining both
feature extractor and x
"""
cnn_model = Model(
    inputs=feature_extractor.input,
    outputs=prediction_layer)
cnn_model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# print(cnn_model.summary())


"""
Plot Convolutional Feature Extractor 
    Architecture
"""
# plot_model(
#     cnn_model,
#     to_file='ConvFeatureExtractorArchitecture.png',
#     show_shapes=True,
#     show_layer_names=True)


"""
MODEL TRAINING
"""
history = cnn_model.fit(
    X_train, y_train_one_hot,
    epochs=50,
    validation_data=(X_test, y_test_one_hot))


"""
PLOT TRAINING AND VALIDATION ACCURACY AND
        LOSS AT EACH EPOCH

PLOT TRAINING AND VALIDATION ACCURACY AND
        LOSS AT EACH EPOCH
"""
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
# plt.show()


"""
CONFUSION MATRIX
CONFUSION MATRIX
CONFUSION MATRIX
"""
prediction_NN = cnn_model.predict(
    X_test)
prediction_NN = np.argmax(
    prediction_NN,
    axis=-1)
prediction_NN = label_encoder.inverse_transform(
    prediction_NN)


cm = confusion_matrix(test_labels, prediction_NN)
# print(cm)
sns.heatmap(cm, annot=True)
# plt.show()


"""
Now, let us use features from 
        convolutional network for RF
"""

X_for_RF = feature_extractor.predict(
    X_train)  # This is out X input to RF

RF_model = RandomForestClassifier(
    n_estimators=50,
    random_state=2023)

"""
Train the model on training data
For sklearn no one hot encoding
"""
RF_model.fit(X_for_RF, y_train)

# Send test data through same feature extractor process
X_test_feature = feature_extractor.predict(X_test)

# Now predict using the trained RF model.
prediction_RF = RF_model.predict(X_test_feature)

# Inverse label encoder transform to get
# original label back.
prediction_RF = label_encoder.inverse_transform(
    prediction_RF)

# Print overall accuracy
print("Accuracy = ", metrics.accuracy_score(
    test_labels, prediction_RF))

# Confusion Matrix - verify accuracy of each class
cm = confusion_matrix(test_labels, prediction_RF)
# print(cm)
sns.heatmap(cm, annot=True)
# plt.show()
