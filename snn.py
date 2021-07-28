import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from keras import layers,models
from keras.layers import Dropout
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

# Lateral
# Open the data and labels
pkl_file = open('Lateral_data_no_CNN.pkl', 'rb')
train_data= pickle.load(pkl_file)
pkl_file.close()
pkl_file = open('Lateral_labels_no_CNN.pkl', 'rb')
labels= pickle.load(pkl_file)
pkl_file.close()

# Split the data and shuffle it. 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.2)
X_train=tf.convert_to_tensor(X_train)
X_test=tf.convert_to_tensor(X_test)
y_train=tf.convert_to_tensor(y_train)
y_test=tf.convert_to_tensor(y_test)

# 10% of train set as validation set
x_val = X_train[-204:]
y_val = y_train[-204:]
X_train = X_train[:-50]
y_train = y_train[:-50]


# Define the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(27, 2)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(6, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train and test the model
history=model.fit(X_train, y_train, epochs=400, validation_data=(x_val, y_val))

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=0) # Testing the model
print('Test accuracy:', test_acc)



# Train and validation accuracy through epochs
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(hist['epoch'], hist['accuracy'],
           label='Train accuracy')
  plt.plot(hist['epoch'], hist['val_accuracy'],
           label = 'Val accuracy')
  plt.ylim([0,1])
  plt.legend()
plot_history(history)


# Confusion matrix
a=model.predict(X_test)
l_predictions=[]
i=0
while i<a.shape[0]:
  l_predictions.append(np.argmax(a[i]))
  i+=1
tf.math.confusion_matrix(y_test,l_predictions)


cm = confusion_matrix(y_true=y_test, y_pred=l_predictions)
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

l=['Initial position','Pole plant','Push','Pole release','Leg push','Sixth Pattern']
classes=np.array(l)

plot_confusion_matrix(cm=cm, classes=classes, title='Confusion Matrix')

model.save('Lateral_sNN.h5')



# Frontal
# Same process as with lateral, but using the frontal data instead
Lateral_data = open('Frontal_data_no_CNN.pkl', 'wb')
pickle.dump(train_data,Lateral_data)
Lateral_data.close()

Lateral_labels = open('Frontal_labels_no_CNN.pkl', 'wb')
pickle.dump(labels,Lateral_labels)
Lateral_labels.close()



X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.2)
X_train=tf.convert_to_tensor(X_train)
X_test=tf.convert_to_tensor(X_test)
y_train=tf.convert_to_tensor(y_train)
y_test=tf.convert_to_tensor(y_test)

x_val = X_train[-201:]
y_val = y_train[-201:]
X_train = X_train[:-201]
y_train = y_train[:-201]

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(27, 2)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(6, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history=model.fit(X_train, y_train, validation_data=(x_val, y_val), epochs=50)

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=0) # Testing the model
print('Test accuracy:', test_acc)



model.summary() # Command to visualise the summary of the model (this has been used for the three models)

a=model.predict(X_test)

l_predictions=[]
i=0
while i<a.shape[0]:
  l_predictions.append(np.argmax(a[i]))
  i+=1

tf.math.confusion_matrix(y_test,l_predictions)

cm = confusion_matrix(y_true=y_test, y_pred=l_predictions)

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

l=['Initial position','Pole plant','Push','Pole release','Leg push','Sixth Pattern']
classes=np.array(l)

plot_confusion_matrix(cm=cm, classes=classes, title='Confusion Matrix')

model.save('Frontal_sNN.h5')

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(hist['epoch'], hist['accuracy'],
           label='Train accuracy')
  plt.plot(hist['epoch'], hist['val_accuracy'],
           label = 'Val accuracy')
  plt.ylim([0,1])
  plt.legend()
plot_history(history)
