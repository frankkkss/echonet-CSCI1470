from model import FrameSelect, Unet
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
    


model = FrameSelect((112, 112, 128, 1))
model.build(input_shape= (100, 112, 112, 128, 1))
model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

model = Unet((112, 112, 1))
model.build(input_shape= (100, 112, 112, 1))
model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy']) #cambiar loss y puede que metric
model.summary()


