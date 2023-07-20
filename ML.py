#this piece of code is going to train a CNN to classify images of polynomials into their leading orders,
# it will consist of 2 layers, the first layer will have 128 neurons, the second layer will have 10 neurons

from PIL import Image
import numpy as np
import os
import random
import tensorflow as tf


#Step 0: Image processing
#this function convertsa colored image to a black and white one,removing the color channel
def rgb_to_grayscale(rgb_array):
    # Ensure the input array has the correct shape
    assert rgb_array.shape[-1] == 4, "Input array must have 4 channels (R, G, B, Alpha)"

    # Convert RGB to grayscale using the luminosity method
    # Grayscale value = 0.21 * R + 0.72 * G + 0.07 * B
    grayscale_array = np.dot(rgb_array[:, :, :3], [0.21, 0.72, 0.07])

    return grayscale_array

#this function downses the image by a factor of shrinking_factor
def downsize_image(image, shrinking_factor):
    # Get the original width and height of the image
    original_width, original_height = image.size
    #print("hello")

    # Calculate the new width and height after downsizing
    new_width = int(original_width * shrinking_factor)
    new_height = int(original_height * shrinking_factor)

    # Resize the image using the calculated new dimensions
    downsized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    return downsized_image


#Step 1: Data Preperation
#this part of the code will go into the subdirectory /images/ and convert each image into a numpy array, and then append it to a list, then append the images name to a seperate list
# it will then split the data into training and testing sets, and then it will convert the training and testing sets into numpy arrays
def image_loader():
    #this function will load all the images in the images folder, and convert them to a numpy array
    #it will then append the numpy array to a list, and the name of the image to a seperate list
    #it will then return both lists
    images = []
    labels = []
    for filename in os.listdir("images"):
        if filename.endswith(".png"):
            #print(filename[1])
            img = Image.open("images/" + filename)
            img = downsize_image(img, 1)
            img = np.array(img)
            #print(np.shape(img))
            img = rgb_to_grayscale(img)
            #print(np.shape(img))
            img = img.flatten()
            images.append(img)
            #print(np.shape(img))    
            labels.append(filename[1])
            continue
        else:
            continue
    return images, labels

def make_y_onehot(y_data):
    y_data_int = np.array(y_data, dtype=np.int32)
    num_classes = 10
    y_data_onehot = tf.keras.utils.to_categorical(y_data_int, num_classes)
    return y_data_onehot


def split_and_shuffle_data(x_data, y_data, test_ratio=0.2):
    # Combine x_data and y_data into pairs and shuffle them together
    combined_data = list(zip(x_data, y_data))
    random.shuffle(combined_data)

    # Calculate the index to split the data into training and testing sets
    split_index = int(len(combined_data) * (1 - test_ratio))

    # Split the shuffled data back into x_train, y_train, x_test, and y_test
    x_train, y_train = zip(*combined_data[:split_index])
    x_test, y_test = zip(*combined_data[split_index:])

    return list(x_train), list(y_train), list(x_test), list(y_test)

def convert_to_numpy(x_train, y_train, x_test, y_test):
    # Convert the data to numpy arrays and return the numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test

def create_model(input_size):
    model = tf.keras.Sequential([
        # Input layer with the given input size
        tf.keras.layers.InputLayer(input_shape=(input_size,)),
        
        # First Dense layer to reduce the dimension to 65
        tf.keras.layers.Dense(65, activation='relu'),
        
        # Second Dense layer to reduce the dimension to 10
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model
#step 3: Model Compilation
#this part of the code will compile the model, and apply the specified optimizer and loss function
def compile_model(model):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Display the summary of the model
    model.summary()
    return model

#step 4: Model Training
#this part of the code will train the model, and then save the model
def train_model(model, x_train, y_train, x_test, y_test):
    # Train the model for 10 epochs using the training set
    model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

    # Save the model
    model.save('model.h5')

x_data, y_data = image_loader()
y_data = make_y_onehot(y_data)
x_train,y_train,x_test,y_test = split_and_shuffle_data(x_data,y_data)
x_train, y_train, x_test, y_test = convert_to_numpy(x_train, y_train, x_test, y_test)
print(np.shape(x_train[0])[0])
print(np.shape(y_train))
print(y_train)
model = create_model(np.shape(x_train[0])[0])
model = compile_model(model)
train_model(model, x_train, y_train, x_test, y_test)

