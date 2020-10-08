#
# Run using command: streamlit run TrainApplication.py
#


#
# Libraries
#

# NumPy
import numpy as np # linear algebra
# Pandas
import pandas as pd
# Matplotlib is a graphing library
import matplotlib.pyplot as plt
#import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
#streamlit
import streamlit as st
from streamlit import components
# OS
import os
# mean_absolute_error
from sklearn.metrics import mean_absolute_error
#train_test_split
from sklearn.model_selection import train_test_split
# TensorFlow is an open source machine learning library
import tensorflow as tf
# Keras is TensorFlow's high-level API for deep learning
from tensorflow import keras
# Math is Python's math library
import math




#####################################################################################


#
# Functions
#


#####################################################################################

st.title("TinyML for uC: train and implement a sin() signal")

#
# Configuration
#
# Define paths to model files
import os
MODELS_DIR = 'models/'
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
MODEL_TF = MODELS_DIR + 'model.pb'
MODEL_NO_QUANT_TFLITE = MODELS_DIR + 'model_no_quant.tflite'
MODEL_TFLITE = MODELS_DIR + 'model.tflite'
MODEL_TFLITE_MICRO = MODELS_DIR + 'model.cc'

#
# Generate Data
#
"""
### Generate Data

The code in the following cell will generate a set of random x values, calculate their sine values, and display them on a graph.
"""
# Number of sample datapoints
SAMPLES = 1000

# Generate a uniformly distributed set of random numbers in the range from
# 0 to 2π, which covers a complete sine wave oscillation
x_values = np.random.uniform(low=0, high=2*math.pi, size=SAMPLES).astype(np.float32)

# Shuffle the values to guarantee they're not in order
np.random.shuffle(x_values)
#print(x_values) # uncomment to see values

# Calculate the corresponding sine values
y_values = np.sin(x_values).astype(np.float32)

# Plot our data. The 'b.' argument tells the library to print blue dots.
fig, ax = plt.subplots()
ax.plot(x_values, y_values, 'b.')
st.pyplot(fig)

"""
Add noise:
"""
# Add a small random number to each y value
y_values += 0.1 * np.random.randn(*y_values.shape)

# Plot our data
fig, ax = plt.subplots()
ax.plot(x_values, y_values, 'b.')
st.pyplot(fig)

"""
Split the data:
"""
#
train_slider = st.slider('train[%]', 0, 100, 60)
test_slider = st.slider('test[%]', 0, (100-train_slider), int(((100-train_slider)/2)))
validation_slider = st.slider('validation[%]', 0, (100-train_slider-test_slider), (100-train_slider-test_slider))


# We'll use 60% of our data for training and 20% for testing. The remaining 20%
# will be used for validation. Calculate the indices of each section.
#TRAIN_SPLIT =  int(0.6 * SAMPLES)
#TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)
TRAIN_SPLIT =  int(train_slider/100 * SAMPLES)
TEST_SPLIT = int(test_slider/100 * SAMPLES + TRAIN_SPLIT)

# Use np.split to chop our data into three parts.
# The second argument to np.split is an array of indices where the data will be
# split. We provide two indices, so the data will be divided into three chunks.
x_train, x_test, x_validate = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_test, y_validate = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

# Double check that our splits add up correctly
assert (x_train.size + x_validate.size + x_test.size) ==  SAMPLES

"""
Plot:
"""
# Plot the data in each partition in different colors:
fig, ax = plt.subplots()
ax.plot(x_train, y_train, 'b.', label="Train")
ax.plot(x_test, y_test, 'r.', label="Test")
ax.plot(x_validate, y_validate, 'y.', label="Validate")
ax.legend()
st.pyplot(fig)


#
# Training the first model
#
"""
### Training the first model

We're going to build a simple neural network model that will take an input value (in this case, x) and use it to predict a numeric output value (the sine of x). This type of problem is called a regression. It will use layers of neurons to attempt to learn any patterns underlying the training data, so it can make predictions.

To begin with, we'll define two layers. The first layer takes a single input (our x value) and runs it through 16 neurons. Based on this input, each neuron will become activated to a certain degree based on its internal state (its weight and bias values). A neuron's degree of activation is expressed as a number.

The activation numbers from our first layer will be fed as inputs to our second layer, which is a single neuron. It will apply its own weights and bias to these inputs and calculate its own activation, which will be output as our y value.
"""
neurons_input = st.number_input('Neurons', min_value=1, max_value=128, value=16)
epochs_input = st.number_input('Epochs', min_value=1, max_value=5000, value=100)
batch_size_input = st.number_input('Batch size', min_value=1, max_value=TRAIN_SPLIT, value=16)

# We'll use Keras to create a simple model architecture
model_1 = tf.keras.Sequential()

# First layer takes a scalar input and feeds it through 16 "neurons". The
# neurons decide whether to activate based on the 'relu' activation function.
model_1.add(keras.layers.Dense(neurons_input, activation='relu', input_shape=(1,)))

# Final layer is a single neuron, since we want to output a single value
model_1.add(keras.layers.Dense(1))

# Compile the model using a standard optimizer and loss function for regression
model_1.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Summary
sum = model_1.summary()

#Parameters:#
#
#    first layer: 16 neurons - 16 connections
#    second layer: 1 neuron - 16 connections
#    biases: 16 first layer + 1 second layer = 17
#    total 49 parameters
"""
#### Training:
"""
if st.checkbox('Train Model 1!'):
    # Train the model on our training data while validating on our validation set
    history_1 = model_1.fit(x_train, y_train, epochs=epochs_input, batch_size=batch_size_input,
                        validation_data=(x_validate, y_validate))
    loss, mae = model_1.evaluate(x_train, y_train)
    st.write("Loss: ", loss)
    st.write("MAE: ", mae)
    print("Loss: ", loss)
    st.write('Training done!!!')

    """
    #### Plot metrics:

    ##### Mean Squared Error

    During training, the model's performance is constantly being measured against both our training data and the validation data that we set aside earlier. Training produces a log of data that tells us how the model's performance changed over the course of the training process.

    The following cells will display some of that data in a graphical form:
    """
    # Draw a graph of the loss, which is the distance between
    # the predicted and actual values during training and validation.
    loss = history_1.history['loss']
    val_loss = history_1.history['val_loss']

    epochs = range(1, len(loss) + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs, loss, 'g.', label='Training loss')
    ax.plot(epochs, val_loss, 'b', label='Validation loss')
    ax.legend()
    st.pyplot(fig)

    """
    The graph shows the loss (or the difference between the model's predictions and the actual data) for each epoch. There are several ways to calculate loss, and the method we have used is mean squared error. There is a distinct loss value given for the training and the validation data.

    As we can see, the amount of loss rapidly decreases over the first 25 epochs, before flattening out. This means that the model is improving and producing more accurate predictions!

    Our goal is to stop training when either the model is no longer improving, or when the training loss is less than the validation loss, which would mean that the model has learned to predict the training data so well that it can no longer generalize to new data.

    To make the flatter part of the graph more readable, let's skip the first 10% epochs:
    """

    # Exclude the first few epochs so the graph is easier to read
    SKIP = int(epochs_input * 0.1)
    fig, ax = plt.subplots()
    ax.plot(epochs[SKIP:], loss[SKIP:], 'g.', label='Training loss')
    ax.plot(epochs[SKIP:], val_loss[SKIP:], 'b.', label='Validation loss')
    ax.legend()
    st.pyplot(fig)

    """
    From the plot, we can see that loss continues to reduce until around 200 epochs, at which point it is mostly stable. This means that there's no need to train our network beyond 200 epochs.

    However, we can also see that the lowest loss value is still around 0.155. This means that our network's predictions are off by an average of ~15%. In addition, the validation loss values jump around a lot, and is sometimes even higher.
    """

    """
    ##### Mean Absolute Error

    To gain more insight into our model's performance we can plot some more data. This time, we'll plot the mean absolute error, which is another way of measuring how far the network's predictions are from the actual numbers:
    """
    # Draw a graph of mean absolute error, which is another way of
    # measuring the amount of error in the prediction.
    mae = history_1.history['mae']
    val_mae = history_1.history['val_mae']
    fig, ax = plt.subplots()
    ax.plot(epochs[SKIP:], mae[SKIP:], 'g.', label='Training MAE')
    ax.plot(epochs[SKIP:], val_mae[SKIP:], 'b.', label='Validation MAE')
    ax.legend()
    st.pyplot(fig)

    """
    This graph of mean absolute error tells another story. We can see that training data shows consistently lower error than validation data, which means that the network may have overfit, or learned the training data so rigidly that it can't make effective predictions about new data.

    In addition, the mean absolute error values are quite high, ~0.305 at best, which means some of the model's predictions are at least 30% off. A 30% error means we are very far from accurately modelling the sine wave function.
    """

    """
    ##### Actual vs Predicted Outputs

    To get more insight into what is happening, let's check its predictions against the test dataset we set aside earlier:
    """
    # Calculate and print the loss on our test dataset
    loss = model_1.evaluate(x_test, y_test)

    # Make predictions based on our test dataset
    predictions = model_1.predict(x_test)

    # Graph the predictions against the actual values
    fig, ax = plt.subplots()
    #plt.title('Comparison of predictions and actual values')
    ax.plot(x_test, y_test, 'b.', label='Actual')
    ax.plot(x_test, predictions, 'r.', label='Predicted')
    ax.legend()
    st.pyplot(fig)


    """
    Oh dear! The graph makes it clear that our network has learned to approximate the sine function in a very limited way.

    The rigidity of this fit suggests that the model does not have enough capacity to learn the full complexity of the sine wave function, so it's only able to approximate it in an overly simplistic way. By making our model bigger, we should be able to improve its performance.
    """

"""
### Training a Larger Model

To make our model bigger, let's add an additional layer of neurons. The following cell redefines our model in the same way as earlier, but with 16 neurons in the first layer and an additional layer of 16 neurons in the middle:

"""
neurons_input1 = st.number_input('Neurons_1', min_value=1, max_value=128, value=16)
neurons_input2 = st.number_input('Neurons_2', min_value=1, max_value=128, value=16)
epochs_input1 = st.number_input('Epochs_1', min_value=1, max_value=5000, value=500)
batch_size_input1 = st.number_input('Batch size_1', min_value=1, max_value=TRAIN_SPLIT, value=16)

#
model_2 = tf.keras.Sequential()

# First layer takes a scalar input and feeds it through 16 "neurons". The
# neurons decide whether to activate based on the 'relu' activation function.
model_2.add(keras.layers.Dense(neurons_input1, activation='relu', input_shape=(1,)))

# The new second layer may help the network learn more complex representations
model_2.add(keras.layers.Dense(neurons_input2, activation='relu'))

# Final layer is a single neuron, since we want to output a single value
model_2.add(keras.layers.Dense(1))

# Compile the model using a standard optimizer and loss function for regression
model_2.compile(optimizer='adam', loss='mse', metrics=['mae'])

"""
#### Training
"""
if st.checkbox('Train Model 2!'):
    history_2 = model_2.fit(x_train, y_train, epochs=epochs_input1, batch_size=batch_size_input1,
                    validation_data=(x_validate, y_validate))

    loss, mae = model_2.evaluate(x_train, y_train)
    st.write("Loss: ", loss)
    st.write("MAE: ", mae)
    print("Loss: ", loss)
    st.write('Training done!!!')

    """
    #### Plot Metrics

    ##### Loss:
    """

    # Draw a graph of the loss, which is the distance between
    # the predicted and actual values during training and validation.
    loss = history_2.history['loss']
    val_loss = history_2.history['val_loss']

    epochs = range(1, len(loss) + 1)

    # Exclude the first few epochs so the graph is easier to read
    SKIP = int(epochs_input1 * 0.1)        
    fig, ax = plt.subplots()
    ax.plot(epochs[SKIP:], loss[SKIP:], 'g.', label='Training loss')
    ax.plot(epochs[SKIP:], val_loss[SKIP:], 'b.', label='Validation loss')
    st.pyplot(fig)

    """
    ##### MAE:
    """
    # Draw a graph of mean absolute error, which is another way of
    # measuring the amount of error in the prediction.
    mae = history_2.history['mae']
    val_mae = history_2.history['val_mae']
    fig, ax = plt.subplots()
    ax.plot(epochs[SKIP:], mae[SKIP:], 'g.', label='Training MAE')
    ax.plot(epochs[SKIP:], val_mae[SKIP:], 'b.', label='Validation MAE')
    st.pyplot(fig)

    """
    #### Predictions:
    """
    # Calculate and print the loss on our test dataset
    loss = model_2.evaluate(x_test, y_test)

    # Make predictions based on our test dataset
    predictions = model_2.predict(x_test)

    # Graph the predictions against the actual values
    fig, ax = plt.subplots()
    ax.plot(x_test, y_test, 'b.', label='Actual')
    ax.plot(x_test, predictions, 'r.', label='Predicted')
    ax.legend()
    st.pyplot(fig)

    """
    Much better! The evaluation metrics we printed show that the model has a low loss and MAE on the test data, and the predictions line up visually with our data fairly well.

    The model isn't perfect; its predictions don't form a smooth sine curve. For instance, the line is almost straight when x is between 4.2 and 5.2. If we wanted to go further, we could try further increasing the capacity of the model, perhaps using some techniques to defend from overfitting.

    However, an important part of machine learning is knowing when to quit, and this model is good enough for our use case - which is to make some LEDs blink in a pleasing pattern.
    """

    """
    ### Generate a TensorFlow Lite Model

    #### Generate Models with or without Quantization

    We now have an acceptably accurate model. We'll use the [TensorFlow Lite Converter](https://www.tensorflow.org/lite/convert) to convert the model into a special, space-efficient format for use on memory-constrained devices.

    Since this model is going to be deployed on a microcontroller, we want it to be as tiny as possible! One technique for reducing the size of models is called [quantization](https://www.tensorflow.org/lite/performance/post_training_quantization) while converting the model. It reduces the precision of the model's weights, and possibly the activations (output of each layer) as well, which saves memory, often without much impact on accuracy. Quantized models also run faster, since the calculations required are simpler.

    *Note: Currently, TFLite Converter produces TFlite models with float interfaces (input and output ops are always float). This is a blocker for users who require TFlite models with pure int8 or uint8 inputs/outputs. Refer to https://github.com/tensorflow/tensorflow/issues/38285*

    In the following cell, we'll convert the model twice: once with quantization, once without.
    """
    # Convert the model to the TensorFlow Lite format without quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model_2)
    model_no_quant_tflite = converter.convert()

    # # Save the model to disk
    open(MODEL_NO_QUANT_TFLITE, "wb").write(model_no_quant_tflite)

    # Convert the model to the TensorFlow Lite format with quantization
    def representative_dataset():
        for i in range(500):
            yield([x_train[i].reshape(1, 1)])
    # Set the optimization flag.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Enforce full-int8 quantization (except inputs/outputs which are always float)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Provide a representative dataset to ensure we quantize correctly.
    converter.representative_dataset = representative_dataset
    model_tflite = converter.convert()

    # Save the model to disk
    open(MODEL_TFLITE, "wb").write(model_tflite)

    """
    #### Compare Model Sizes
    """
    model_no_quant_size = os.path.getsize(MODEL_NO_QUANT_TFLITE)
    st.write("Model is %d bytes" % model_no_quant_size)
    model_size = os.path.getsize(MODEL_TFLITE)
    st.write("Quantized model is %d bytes" % model_size)
    difference = model_no_quant_size - model_size
    st.write("Difference is %d bytes" % difference)

    """
    Our quantized model is only ~200 bytes smaller than the original version, which only a tiny reduction in size! At around 2.5 kilobytes, this model is already so small that the weights make up only a small fraction of the overall size, meaning quantization has little effect.

    More complex models have many more weights, meaning the space saving from quantization will be much higher, approaching 4x for most sophisticated models.

    Regardless, our quantized model will take less time to execute than the original version, which is important on a tiny microcontroller!
    """

    """
    #### Test the Models

    To prove these models are still accurate after conversion and quantization, we'll use both of them to make predictions and compare these against our test results:
    """
    # Instantiate an interpreter for each model
    model_no_quant = tf.lite.Interpreter(MODEL_NO_QUANT_TFLITE)
    model = tf.lite.Interpreter(MODEL_TFLITE)

    # Allocate memory for each model
    model_no_quant.allocate_tensors()
    model.allocate_tensors()

    # Get the input and output tensors so we can feed in values and get the results
    model_no_quant_input = model_no_quant.tensor(model_no_quant.get_input_details()[0]["index"])
    model_no_quant_output = model_no_quant.tensor(model_no_quant.get_output_details()[0]["index"])
    model_input = model.tensor(model.get_input_details()[0]["index"])
    model_output = model.tensor(model.get_output_details()[0]["index"])

    # Create arrays to store the results
    model_no_quant_predictions = np.empty(x_test.size)
    model_predictions = np.empty(x_test.size)

    # Run each model's interpreter for each value and store the results in arrays
    for i in range(x_test.size):
        model_no_quant_input().fill(x_test[i])
        model_no_quant.invoke()
        model_no_quant_predictions[i] = model_no_quant_output()[0]

        model_input().fill(x_test[i])
        model.invoke()
        model_predictions[i] = model_output()[0]

    # See how they line up with the data
    fig, ax = plt.subplots()
    ax.plot(x_test, y_test, 'bo', label='Actual values')
    ax.plot(x_test, predictions, 'ro', label='Original predictions')
    ax.plot(x_test, model_no_quant_predictions, 'bx', label='Lite predictions')
    ax.plot(x_test, model_predictions, 'gx', label='Lite quantized predictions')
    ax.legend()
    st.pyplot(fig)

    """
    We can see from the graph that the predictions for the original model, the converted model, and the quantized model are all close enough to be indistinguishable. This means that our quantized model is ready to use!
    """

    """
    ### Generate a TensorFlow Lite for Microcontrollers Model

    Convert the TensorFlow Lite quantized model into a C source file that can be loaded by TensorFlow Lite for Microcontrollers.
    """
    # Install xxd if it is not available
    #apt-get update && apt-get -qq install xxd
    # Convert to a C source file
    #xxd -i {MODEL_TFLITE} > {MODEL_TFLITE_MICRO}
    # Update variable names
    #REPLACE_TEXT = MODEL_TFLITE.replace('/', '_').replace('.', '_')
    #sed -i 's/'{REPLACE_TEXT}'/g_model/g' {MODEL_TFLITE_MICRO}


    """
    ### Deploy to a Microcontroller

    Follow the instructions in the [hello_world](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/hello_world) README.md for [TensorFlow Lite for MicroControllers](https://www.tensorflow.org/lite/microcontrollers/overview) to deploy this model on a specific microcontroller.

    **Reference Model:** If you have not modified this notebook, you can follow the instructions as is, to deploy the model. Refer to the [`hello_world/train/models`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/hello_world/train/models) directory to access the models generated in this notebook.

    **New Model:** If you have generated a new model, then update the values assigned to the variables defined in [`hello_world/model.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/hello_world/model.cc) with values displayed after running the following cell.
    """
    # Print the C source file
    #cat {MODEL_TFLITE_MICRO}