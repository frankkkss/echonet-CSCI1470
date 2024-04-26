from model import FrameSelect, Unet

model = FrameSelect((112, 112, 128, 1))
model.build(input_shape= (100, 112, 112, 128, 1))
model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

model = Unet((112, 112, 1))
model.build(input_shape= (100, 112, 112, 1))
model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy']) #cambiar loss y puede que metric
model.summary()