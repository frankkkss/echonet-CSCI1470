from model import FrameSelect

model = FrameSelect((112, 112, 128, 1))
model.build(input_shape= (100, 112, 112, 128, 1))
model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()