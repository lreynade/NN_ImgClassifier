# NN_ImgClassifier

This is a CodeAcademy project. The goal of the project is to build a model that can help doctors diagnose
illnesses that affects a patient's lungs. Three classes charaterize the data points: COVID, Normal, and Pneumonia.
The training and test sets contain 251 and 66 images, respectively. All the images are black and white.

The model has six hidden layers but the number of trainable parameters is 340.
Some of the non-trainable parameters are:

* Learning_rate = 0.008
* steps_per_epoch = len(training_iterator)/12
* validation_steps = len(valid_iterator)/8
* epochs = 60
* Training batch_size = 32
* Test batch_size = 12

# Remarks 
Getting comparable training and validation results was not easy.

* loss = 0.7605
* categorical_accuracy = 0.6562
* accuracy = 0.8352
* val_loss = 0.7386
* val_cat_accuracy = 0.8333
* val_accuracy = 0.9306
