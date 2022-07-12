import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.metrics as metrics
import seaborn as sns
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pandas as pd
import keras
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, Model, load_model

#reading training and testing files
train = pd.read_csv("C:\\Users\hp\Desktop\\ind project\\project code\\train.csv")
test = pd.read_csv('C:\\Users\\hp\\Desktop\\ind project\\project code\\test.csv')
print(train.head(10))
print(train.shape)
print(test.shape)
#remove the label column from the train dataset
X = train.drop('label' , axis = 1)
y = np.array(train['label'])
#split the data for testing and training
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_shape = (784 , )
num_classes = 10

#normalize values between 0 and 1   
X_train_flattened = X_train/255
X_test_flattened = X_test/255

#model definition
model = keras.Sequential([
    keras.layers.Dense(512, input_shape=(784,), activation='relu'), 
    keras.layers.BatchNormalization(),
    keras.layers.Dense(256 , activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(128 , activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64 , activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(32 , activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(16 , activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10 , activation = 'softmax'),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# fitting and training of model
hist = model.fit(X_train_flattened,y_train,epochs=25, validation_split = 0.2)
plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy from the his object
plt.plot(hist.history['accuracy'], label='train acc')
plt.plot(hist.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

model.evaluate(X_test_flattened,y_test)

y_predicted = model.predict(X_test_flattened)
np.argmax(y_predicted[0])

y_predicted_labels = [np.argmax(i) for i in y_predicted]

cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
print(cm)

plt.figure(figsize=(12, 8))
im=sns.heatmap(cm, annot=True, fmt='d')
plt.show()
X_test = np.array(test)
predictions = np.argmax(model.predict(X_test), axis=-1)
test['Label'] = predictions
test['ImageId'] = list(range(1, 28001 , 1))
test[['ImageId' , 'Label']].to_csv('submission.csv' , index = False)



