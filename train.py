import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from util.my_model import get_model, rescale, load_data
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
vggmodel = get_model()
X_train, y_train = load_data('one_hot_data\pix150.data')
X_train2, y_train2 = load_data('one_hot_data\pix100.data')
X_train = np.concatenate((X_train, X_train2), axis=0)
y_train = np.concatenate((y_train, y_train2), axis=0)
X_train, y_train = shuffle(X_train, y_train)
def split(x, y, ratio):
    x, y = shuffle(x, y)
    x, x_tmp, y, y_tmp = train_test_split(x, y, train_size=ratio, random_state=100)
    return x, y
X_val, y_val = load_data('one_hot_data\pix400.data')
X_val, y_val = split(X_val, y_val, 0.25)
print("Train data shape: ", X_train.shape)
print("Validation data shape: ", X_val.shape)

aug = ImageDataGenerator(rotation_range=90, zoom_range=0.1,
    preprocessing_function=rescale,
	width_shift_range=0.15,
    height_shift_range=0.15,
    vertical_flip=True,
    brightness_range=[0.2,1.5], fill_mode="nearest")

filepath="weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
aug_val = ImageDataGenerator(preprocessing_function=rescale)

def plot_model_accuracy(model_history):
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('model_acc.png')
    plt.show()

def plot_model_loss(model_history):
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('model_loss.png')
    plt.show()

vgghist = vggmodel.fit_generator(
    aug.flow(X_train, y_train, batch_size=32),
    epochs=50, 
    validation_data=aug_val.flow(X_val,y_val, batch_size=32),
    callbacks=callbacks_list)
plot_model_accuracy(vgghist)
plot_model_loss(vgghist)
