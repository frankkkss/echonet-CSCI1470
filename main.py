from model import FrameSelect, Unet
import tensorflow as tf
import argparse
from preprocess import splits 

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oscar", action="store_true")
    # parser.add_argument("--load_weights", action="store_true")
    # parser.add_argument("--batch_size", type=int, default=128)
    # parser.add_argument("--num_epochs", type=int, default=10)
    # parser.add_argument("--latent_size", type=int, default=15)
    # parser.add_argument("--input_size", type=int, default=28 * 28)
    # parser.add_argument("--learning_rate", type=float, default=1e-3)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseArguments()

    mask_train, mask_val, mask_test, ims_train, ims_val, ims_test, vids_train, vids_val, vids_test, labels_sys_train, labels_sys_val, labels_sys_test, labels_dyas_train, labels_dyas_val, labels_dyas_test = splits(args)

    video_size = (112, 112, 128, 1)
    systole = FrameSelect(video_size)
    dyastole = FrameSelect(video_size)

    systole.build(input_shape= (8722, 112, 112, 128, 1))
    dyastole.build(input_shape= (8722, 112, 112, 128, 1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    systole.compile(optimizer= optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    dyastole.compile(optimizer= optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    systole.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # La métrica a monitorizar
        patience=3,  # Cuántas épocas sin mejora antes de detener
        restore_best_weights=True,  # Restaura los mejores pesos encontrados durante el entrenamiento
    )

    ## Trainig the model
    print('Start training')

    systole_history = systole.fit(x= vids_train, 
                                y= labels_sys_train, 
                                batch_size= 64, 
                                epochs= 500, 
                                validation_data=(vids_val, labels_sys_val),
                                validation_batch_size= 64,  
                                callbacks= [early_stopping], 
                                verbose= 2)

    # dyastole_history = systole.fit(x= vids_train, 
                                # y= labels_dyas_train, 
                                # batch_size= 64, 
                                # epochs= 500, 
                                # validation_data=(vids_val, labels_dyas_val),
                                # validation_batch_size= 64, 
                                # callbacks= [early_stopping], 
                                # verbose= 2)

    print(f"Systole train and validation loss: {[systole_history.history['loss'], systole_history.history['val_loss']]} \t Systole train and validation accuracy: {[systole_history.history['accuracy'], systole_history.history['val_accuracy']]} \n")
        # f"dyastole train and validation loss: {[dyastole_history.history['loss'], dyastole_history.history['val_loss']]} \t dyastole train and validation accuracy: {[dyastole_history.history['accuracy'], dyastole_history.history['val_accuracy']]}",
        # f"Epochs systole and dyastole: {[len(systole_history.epochs), len(dyastole_history.epochs)]}")

    ## Test the model
    sys_test_loss, sys_test_acc = systole.evaluate(x= vids_test, y= labels_sys_test, batch_size= 64, verbose= 2)
    # dyas_test_loss, dyas_test_acc = dyastole.evaluate(x= vids_test, y= labels_dyas_test, batch_size= 64, verbose= 2)

    print(f"Systole test loss and accuracy: {[sys_test_loss, sys_test_acc]}") #\n Dyastole test loss and accuracy: {[dyas_test_loss, dyas_test_acc]}")

    systole.save('systole_frame_selector.h5')
    dyastole.save('dyastole_frame_selector.h5')