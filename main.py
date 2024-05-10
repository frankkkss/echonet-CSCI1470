from model import FrameSelect, Unet
from eval import *
import tensorflow as tf
import argparse
from preprocess import splits 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oscar", action="store_true", default=False)                      # Argument to choose the file path for running on CCV
    parser.add_argument("--unet", choices=['load', 'train'])                         # Choice of training the model or just loading for testing
    parser.add_argument("--batch_size", action="store_const", const=64)             # Option of changing the batch size
    args = parser.parse_args()
    return args


## To calculate the ejection fraction from a given set of segmentations
def calc_EF(preds, EF):
    pred_EF = np.zeros(shape=EF.shape)
    for i, n in enumerate(EF):
        # Images come in pairs, but the selected frames might be for Systole of Diastole
        mask_1 = preds[2*i, :, :]
        mask_2 = preds[2*i + 1, :, :]
        # Diastole is the relaxation of the myocardium, so end diastolic volume will the moment in which the left ventricle is biggest
        # Systole is the contraction of the myocardium, so end systolic volume will the moment in which the left ventricle is smallest
        EDV = np.max([np.sum(mask_1), np.sum(mask_2)])
        ESV = np.min([np.sum(mask_1), np.sum(mask_2)])
        # Ejection fraction is just a measurement of the "efficiency" of heart in clearing blood inside
        pred_EF[i] = 100*(EDV - ESV)/EDV
    return pred_EF


if __name__ == "__main__":
    args = parseArguments()
    batch_size = args.batch_size

    # Call for the preprocess
    # Missing arguments were for videos and labels
    mask_train, mask_val, mask_test, ims_train, ims_val, ims_test, _, _, vids_test, _, _, _, _, _, _, EF_test = splits(args)

    ### Frame Selection

    # video_size = (112, 112, 128, 1)
    # systole = FrameSelect(video_size)
    # dyastole = FrameSelect(video_size)

    # systole.build(input_shape= (8722, 112, 112, 128, 1))
    # dyastole.build(input_shape= (8722, 112, 112, 128, 1))

    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # systole.compile(optimizer= optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # dyastole.compile(optimizer= optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # systole.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',                                                # Metric being monitorized
        patience= 15,                                                      # N of epochs  waiting for improvement
        restore_best_weights=True,                                         # Restores weights for the best result in the 
    )

    ## Train the model
    # print('Start training Systole')

    # systole_history = systole.fit(x= vids_train, 
    #                             y= labels_sys_train, 
    #                             batch_size= 64, 
    #                             epochs= 500, 
    #                             validation_data=(vids_val, labels_sys_val), 
    #                             callbacks= [early_stopping], 
    #                             verbose= 1)
    
    # print('Start training Dyastole')

    # dyastole_history = systole.fit(x= vids_train, 
    #                             y= labels_dyas_train, 
    #                             batch_size= 64, 
    #                             epochs= 500, 
    #                             validation_data=(vids_val, labels_dyas_val),
    #                             callbacks= [early_stopping], 
    #                             verbose= 1)

    # print(f"Systole train and validation loss: {[systole_history.history['loss'], systole_history.history['val_loss']]}",
    #     f"dyastole train and validation loss: {[dyastole_history.history['loss'], dyastole_history.history['val_loss']]}")

    # Test the model

    # sys_test_loss, sys_test_acc = systole.evaluate(x= vids_test, y= labels_sys_test, batch_size= 64, verbose= 1)
    # dyas_test_loss, dyas_test_acc = dyastole.evaluate(x= vids_test, y= labels_dyas_test, batch_size= 64, verbose= 1)

    # print(f"Systole test loss and accuracy: {[sys_test_loss, sys_test_acc]}\n Dyastole test loss and accuracy: {[dyas_test_loss, dyas_test_acc]}")

    # Save the model

    # tf.saved_model.save(systole, './models/Systole') 
    # tf.saved_model.save(dyastole, './models/Dyastole') 

    ## U-Net
    

    if args.unet == 'train':

        unet = Unet()

        optimizer_unet = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = dice_coef_loss                                       # Loss defined by dice coefficient, measurement for accuracy of segmentations

        unet.compile(optimizer= optimizer_unet, loss= loss, metrics=['accuracy'])

        ## Train the model
        print('Start training Unet')

        unet_history = unet.fit(x= ims_train, 
                                y= mask_train, 
                                batch_size= batch_size, 
                                epochs= 200, 
                                verbose= 1, 
                                callbacks= [early_stopping],
                                validation_data= (ims_val, mask_val))
    
        print(f"Unet train and validation loss: {[unet_history.history['loss'], unet_history.history['val_loss']]}")

        unet_test_loss, unet_test_acc= unet.evaluate(x= ims_test, y= mask_test, batch_size= batch_size, verbose= 1)

        print(f"Unet test loss and accuracy: {[unet_test_loss, unet_test_acc]}")

        ## Save the model

        unet.save('models/Unet')  

        # Prediction of the segmentation by the U-Net for the selected test images
        predictions = unet.predict(ims_test)

        # Calculation of the ejection fraction for the predicted segmentations
        pred_EF = calc_EF(preds=predictions, EF=EF_test)

        # Display of the results for EF prediction and error
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].boxplot(pred_EF, vert=True)
        axs[0].set_title('Distribution of predicted EF')
        axs[1].boxplot(EF_test, vert=True)
        axs[1].set_title('Distribution of ground-truth EF')
        axs[2].scatter(EF_test, pred_EF)
        axs[2].set_xlabel('Ground-truth EF')
        axs[2].set_ylabel('Predicted EF')
        z = np.polyfit(EF_test, pred_EF, 1)
        p = np.poly1d(z)
        axs[2].plot(EF_test, p(EF_test), 'r--')
        text = f"R^2 = {r2_score(EF_test, pred_EF):0.3f}"
        axs[2].gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,fontsize=14, verticalalignment='top')

        plt.tight_layout()
        plt.show()

        # Display of the predicted segmentations and comparison with ground truth
        num_rows = 3
        num_columns = 3
        fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 5))

        # Compare with labels
        for i, prediction in enumerate(predictions):
            
            axs[i, 0].imshow(ims_test[i], cmap='gray')
            axs[i, 0].set_title(f'Frame to segment {i}')
            axs[i, 0].axis('off')

            axs[i, 1].imshow(prediction)
            axs[i, 1].set_title(f'Prediction {i}')
            axs[i, 1].axis('off')
            
            axs[i, 2].imshow(mask_test[i])
            axs[i, 2].set_title(f'Label {i}')
            axs[i, 2].axis('off')

        plt.tight_layout()
        # plt.savefig('Prediction_label_comparation')
        plt.show()

        ## Test the model    
    elif args.unet == 'load':
        
        # Testing the loading of the model
        unet = tf.keras.models.load_model('models/Unet', custom_objects={'dice_coef_loss': dice_coef_loss})

        # Prediction of the segmentation by the U-Net for one of the videos
        predictions = unet.predict(vids_test[1, :, :, :].transpose([2, 0, 1, 3]).astype(np.float32))

        # Calculation of the ejection fraction for the predicted segmentations
        pred_EF = calc_EF(preds=predictions, EF=EF_test)

        # Display of the predicted segmentation on one of the videos, no ground truth available for those
        fig, axs = plt.subplots(1, 5, figsize=(15, 5))
        for i in range (5):
            axs[i].imshow(vids_test[1, :, :, 10*i], cmap=plt.cm.gray, alpha=0.7)
            axs[i].imshow(predictions[10*i, :, :], alpha=0.3)
            axs[i].set_title(f"Frame {10*i + 1}")

        plt.tight_layout()
        # plt.savefig('vid_ex')
        plt.show()


        # Display of the results for EF prediction and error
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].boxplot(pred_EF, vert=True)
        axs[0].set_title('Distribution of predicted EF')
        axs[1].boxplot(EF_test, vert=True)
        axs[1].set_title('Distribution of ground-truth EF')
        axs[2].scatter(EF_test, pred_EF)
        axs[2].set_xlabel('Ground-truth EF')
        axs[2].set_ylabel('Predicted EF')
        z = np.polyfit(EF_test.squeeze(), pred_EF.squeeze(), 1)
        p = np.poly1d(z)
        axs[2].plot(EF_test, p(EF_test), 'r--')
        text = f"$R^2 = {-r2_score(EF_test, pred_EF):0.3f}$"
        axs[2].set_title(text)

        plt.tight_layout()
        # plt.savefig('EF_prediction')
        plt.show()
       

        # Display of the predicted segmentations and comparison with ground truth
        num_rows = len(predictions)
        num_columns = 3
        fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 5))

        ims_test_uint8 = [(im * 255).astype(np.uint8) for im in ims_test]
        predictions_uint8 = [(pred * 255).astype(np.uint8) for pred in predictions]
        mask_test_uint8 = [(mask * 255).astype(np.uint8) for mask in mask_test]

        # Compare with labels
        for i, prediction in enumerate(predictions):
                
            axs[i, 0].imshow(ims_test[i], cmap='gray')
            axs[i, 0].set_title(f'Frame to segment {i}')
            axs[i, 0].axis('off')

            axs[i, 1].imshow(prediction)
            axs[i, 1].set_title(f'Prediction {i}')
            axs[i, 1].axis('off')
            
            axs[i, 2].imshow(mask_test[i])
            axs[i, 2].set_title(f'Label {i}')
            axs[i, 2].axis('off')

        plt.tight_layout()
        # plt.savefig('Prediction_label_comparation')
        plt.show()