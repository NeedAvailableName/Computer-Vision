import tensorflow as tf
import numpy as np
from util.my_model import get_model, rescale, load_data
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

my_model = get_model()
my_model.load_weights("weight\\weight_1.hdf5")
preprocess = ImageDataGenerator(preprocessing_function=rescale)
X_test, y_test = load_data("one_hot_data\pix_test_200.data")

# loss, acc = my_model.evaluate_generator(preprocess.flow(X_test, y_test, batch_size=32))
loss, acc = my_model.evaluate(X_test, y_test)
print('\nEvaluation on Test Dataset')
print('-' * 20)
print('Accuracy: ', acc)
print('Loss: ', loss)

# y_pred = my_model.predict_generator(preprocess.flow(X_test, y_test, batch_size=32))
y_pred = my_model.predict(X_test)
y_true_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

class_name = np.array(['10K_B', '50K_B', '100K_B', '20K_B', '20K_F', '100K_F', 'none', '10K_F', '50K_F'])
class_accuracies = [accuracy_score(y_true_labels == i, y_pred_labels == i) for i in range(9)]

precision_per_class = precision_score(y_true_labels, y_pred_labels, average=None)
recall_per_class = recall_score(y_true_labels, y_pred_labels, average=None)

conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

df = pd.DataFrame({'Class': class_name,
                   'Accuracy': class_accuracies,
                   'Precision': precision_per_class,
                   'Recall': recall_per_class})

print(df)

plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix_normalized, annot=True, cmap="Blues", fmt=".2f", xticklabels=class_name, yticklabels=class_name)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
