import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
from preprocess import splits

#   To obtain some data points
mask_train, mask_val, mask_test, ims_train, ims_val, ims_test, vids_train, vids_val, vids_test, labels_sys_train, labels_sys_val, labels_sys_test, labels_dyas_train, labels_dyas_val, labels_dyas_test = splits(args= None)

# Load and make a prediction

unet = tf.saved_model.load('./Unet')

predictions = unet.predict(ims_test[:3])


num_rows = len(predictions)
num_columns = 3
fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 5))

# Compare with labels
for i, prediction in enumerate(predictions):
    
    axs[i, 0].imshow(prediction)
    axs[i].set_title(f'Frame to segment {i}')
    axs[i].axis('off')

    axs[i, 1].imshow(prediction)
    axs[i].set_title(f'Prediction {i}')
    axs[i].axis('off')
    
    axs[i, 2].imshow(mask_test[i])
    axs[i].set_title(f'Label {i}')
    axs[i].axis('off')

plt.tight_layout()
plt.savefig('Prediction_label_comparation')
plt.show()
