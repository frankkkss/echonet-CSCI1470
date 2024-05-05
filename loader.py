import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
from preprocess import splits
import argparse
from eval import *  


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oscar", action="store_true", default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseArguments()

    #   To obtain some data points
    mask_train, mask_val, mask_test, ims_train, ims_val, ims_test, _, _, _, _, _, _, _, _, _, frames, EF = splits(args)

    # Load and make a prediction

    unet = tf.keras.models.load_model('models/Unet', custom_objects={'dice_coef_loss': dice_coef_loss})

    print(type(unet))

    predictions = unet.predict(ims_test[27:30])


    num_rows = len(predictions)
    num_columns = 3
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 5))

    ims_test_uint8 = [(im * 255).astype(np.uint8) for im in ims_test]
    predictions_uint8 = [(pred * 255).astype(np.uint8) for pred in predictions]
    mask_test_uint8 = [(mask * 255).astype(np.uint8) for mask in mask_test]

    # Compare with labels
    for i, prediction in enumerate(predictions):
        
        axs[i, 0].imshow(ims_test_uint8[i + 27], cmap='gray')
        axs[i, 0].set_title(f'Frame to segment {i}')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(prediction)
        axs[i, 1].set_title(f'Prediction {i}')
        axs[i, 1].axis('off')
        
        axs[i, 2].imshow(mask_test_uint8[i + 27])
        axs[i, 2].set_title(f'Label {i}')
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('Prediction_label_comparation_loader')
    plt.show()
