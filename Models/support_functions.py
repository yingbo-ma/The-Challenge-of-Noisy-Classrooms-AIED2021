# support functions
import xlrd
import os
from PIL import Image
import numpy as np

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras import regularizers
from keras.layers import concatenate
from keras.utils.vis_utils import plot_model

def read_excel(label_file_path):
    data = xlrd.open_workbook(label_file_path)
    table = data.sheet_by_index(0)

    nrows = table.nrows
    ncols = table.ncols

    excel_list = []
    for row in range(0, nrows):
        for col in range(ncols):
            cell_value = int(table.cell(row, col).value)
            excel_list.append(cell_value)
    return excel_list

def spectrogram_data_prepare(excel_list, spectrogram_data_path, pixel, num_channels, split_ratio):

    class_0_list = []
    class_1_list = []
    class_2_list = []

    class_0_data = []
    class_1_data = []
    class_2_data = []

    for i, j in enumerate(excel_list):
        if j == 0:
            class_0_list.append(i)

    for index in range(len(class_0_list)):
        path = os.path.join(spectrogram_data_path, str(class_0_list[index]) + ".jpg")
        image = Image.open(path).resize((pixel, pixel), Image.ANTIALIAS)
        class_0_data.append(np.asarray(image))

    for i, j in enumerate(excel_list):
        if j == 1:
            class_1_list.append(i)

    for index in range(len(class_1_list)):
        path = os.path.join(spectrogram_data_path, str(class_1_list[index]) + ".jpg")
        image = Image.open(path).resize((pixel, pixel), Image.ANTIALIAS)
        class_1_data.append(np.asarray(image))

    for i, j in enumerate(excel_list):
        if j == 2:
            class_2_list.append(i)

    for index in range(len(class_2_list)):
        path = os.path.join(spectrogram_data_path, str(class_2_list[index]) + ".jpg")
        image = Image.open(path).resize((pixel, pixel), Image.ANTIALIAS)
        class_2_data.append(np.asarray(image))

    indices_data_0 = np.arange(len(class_0_data))
    indices_data_1 = np.arange(len(class_1_data))
    indices_data_2 = np.arange(len(class_2_data))

    np.random.shuffle(indices_data_0)
    np.random.shuffle(indices_data_1)
    np.random.shuffle(indices_data_2)

    shuffled_list_0 = indices_data_0.tolist()
    shuffled_list_1 = indices_data_1.tolist()
    shuffled_list_2 = indices_data_2.tolist()

    Class_0_Train_Num = int(len(class_0_data) * split_ratio)
    Class_1_Train_Num = int(len(class_1_data) * split_ratio)
    Class_2_Train_Num = int(len(class_2_data) * split_ratio)

    class_0_train_data = []
    class_0_test_data = []

    class_1_train_data = []
    class_1_test_data = []

    class_2_train_data = []
    class_2_test_data = []

    for i in range(0, Class_0_Train_Num):
        class_0_train_data.append(class_0_data[shuffled_list_0[i]])
    for i in range(Class_0_Train_Num, len(class_0_data)):
        class_0_test_data.append(class_0_data[shuffled_list_0[i]])

    class_0_train_data = np.reshape(class_0_train_data, (-1, pixel, pixel, num_channels))
    class_0_test_data = np.reshape(class_0_test_data, (-1, pixel, pixel, num_channels))

    for i in range(0, Class_1_Train_Num):
        class_1_train_data.append(class_1_data[shuffled_list_1[i]])
    for i in range(Class_1_Train_Num, len(class_1_data)):
        class_1_test_data.append(class_1_data[shuffled_list_1[i]])

    class_1_train_data = np.reshape(class_1_train_data, (-1, pixel, pixel, num_channels))
    class_1_test_data = np.reshape(class_1_test_data, (-1, pixel, pixel, num_channels))

    for i in range(0, Class_2_Train_Num):
        class_2_train_data.append(class_2_data[shuffled_list_2[i]])
    for i in range(Class_2_Train_Num, len(class_2_data)):
        class_2_test_data.append(class_2_data[shuffled_list_2[i]])

    class_2_train_data = np.reshape(class_2_train_data, (-1, pixel, pixel, num_channels))
    class_2_test_data = np.reshape(class_2_test_data, (-1, pixel, pixel, num_channels))

    train_data = np.concatenate((class_0_train_data, class_1_train_data, class_2_train_data), axis=0)
    test_data = np.concatenate((class_0_test_data, class_1_test_data, class_2_test_data), axis=0)

    train_label_list = [[1, 0, 0]]
    for count in range(len(class_0_train_data) - 1):
        train_label_list.append([1, 0, 0])
    for count in range(len(class_1_train_data)):
        train_label_list.append([0, 1, 0])
    for count in range(len(class_2_train_data)):
        train_label_list.append([0, 0, 1])
    train_label = np.asarray(train_label_list)

    test_label_list = [[1, 0, 0]]
    for count in range(len(class_0_test_data) - 1):
        test_label_list.append([1, 0, 0])
    for count in range(len(class_1_test_data)):
        test_label_list.append([0, 1, 0])
    for count in range(len(class_2_test_data)):
        test_label_list.append([0, 0, 1])
    test_label = np.asarray(test_label_list)

    print("data_train.shape: ", train_data.shape)
    print("data_test.shape: ", test_data.shape)
    print("label_test.shape: ", test_label.shape)

    print("shuffled list 0: ", shuffled_list_0)
    print("shuffled list 1: ", shuffled_list_1)
    print("shuffled list 2: ", shuffled_list_2)

    print("length of shuffled list 0: ", len(shuffled_list_0))
    print("length of shuffled list 1: ", len(shuffled_list_1))
    print("length of shuffled list 2: ", len(shuffled_list_2))

    return train_data, train_label, test_data, test_label, shuffled_list_0, shuffled_list_1, shuffled_list_2

def optical_flow_data_prepare(excel_list, optical_flow_data_path, pixel, num_channels, split_ratio, shuffled_list_0, shuffled_list_1, shuffled_list_2):
    class_0_list = []
    class_1_list = []
    class_2_list = []

    class_0_data = []
    class_1_data = []
    class_2_data = []

    for i, j in enumerate(excel_list):
        if j == 0:
            class_0_list.append(i)

    for index in range(len(class_0_list)):
        path = os.path.join(optical_flow_data_path, str(class_0_list[index]) + ".jpg")
        image = Image.open(path).resize((pixel, pixel), Image.ANTIALIAS)
        class_0_data.append(np.asarray(image))

    for i, j in enumerate(excel_list):
        if j == 1:
            class_1_list.append(i)

    for index in range(len(class_1_list)):
        path = os.path.join(optical_flow_data_path, str(class_1_list[index]) + ".jpg")
        image = Image.open(path).resize((pixel, pixel), Image.ANTIALIAS)
        class_1_data.append(np.asarray(image))

    for i, j in enumerate(excel_list):
        if j == 2:
            class_2_list.append(i)

    for index in range(len(class_2_list)):
        path = os.path.join(optical_flow_data_path, str(class_2_list[index]) + ".jpg")
        image = Image.open(path).resize((pixel, pixel), Image.ANTIALIAS)
        class_2_data.append(np.asarray(image))

    Class_0_Train_Num = int(len(class_0_data) * split_ratio)
    Class_1_Train_Num = int(len(class_1_data) * split_ratio)
    Class_2_Train_Num = int(len(class_2_data) * split_ratio)

    class_0_train_data = []
    class_0_test_data = []

    class_1_train_data = []
    class_1_test_data = []

    class_2_train_data = []
    class_2_test_data = []

    for i in range(0, Class_0_Train_Num):
        class_0_train_data.append(class_0_data[shuffled_list_0[i]])
    for i in range(Class_0_Train_Num, len(class_0_data)):
        class_0_test_data.append(class_0_data[shuffled_list_0[i]])

    class_0_train_data = np.reshape(class_0_train_data, (-1, pixel, pixel, num_channels))
    class_0_test_data = np.reshape(class_0_test_data, (-1, pixel, pixel, num_channels))

    for i in range(0, Class_1_Train_Num):
        class_1_train_data.append(class_1_data[shuffled_list_1[i]])
    for i in range(Class_1_Train_Num, len(class_1_data)):
        class_1_test_data.append(class_1_data[shuffled_list_1[i]])

    class_1_train_data = np.reshape(class_1_train_data, (-1, pixel, pixel, num_channels))
    class_1_test_data = np.reshape(class_1_test_data, (-1, pixel, pixel, num_channels))

    for i in range(0, Class_2_Train_Num):
        class_2_train_data.append(class_2_data[shuffled_list_2[i]])
    for i in range(Class_2_Train_Num, len(class_2_data)):
        class_2_test_data.append(class_2_data[shuffled_list_2[i]])

    class_2_train_data = np.reshape(class_2_train_data, (-1, pixel, pixel, num_channels))
    class_2_test_data = np.reshape(class_2_test_data, (-1, pixel, pixel, num_channels))

    train_data = np.concatenate((class_0_train_data, class_1_train_data, class_2_train_data), axis=0)
    test_data = np.concatenate((class_0_test_data, class_1_test_data, class_2_test_data), axis=0)

    print("data_train.shape: ", train_data.shape)
    print("data_test.shape: ", test_data.shape)

    print("shuffled list 0: ", shuffled_list_0)
    print("shuffled list 1: ", shuffled_list_1)
    print("shuffled list 2: ", shuffled_list_2)

    print("length of shuffled list 0: ", len(shuffled_list_0))
    print("length of shuffled list 1: ", len(shuffled_list_1))
    print("length of shuffled list 2: ", len(shuffled_list_2))

    return train_data, test_data

def define_model(input_shape, n_classes):

    in_image = Input(shape=input_shape)

    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(in_image)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.15)(fe)

    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.15)(fe)

    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.15)(fe)

    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.15)(fe)

    fe = Flatten()(fe)

    c_out_layer = Dense(n_classes, activation="softmax", kernel_regularizer=regularizers.l2(0.01))(fe)
    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

    c_model.summary()

    return c_model

def multiple_inputs_model(input_shape_A, input_shape_B, input_shape_C, n_classes):
    in_image_A = Input(shape=input_shape_A)
    in_image_B = Input(shape=input_shape_B)
    in_image_C = Input(shape=input_shape_C)

    # the first branch operates on the first input
    fe_A = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(in_image_A)
    fe_A = BatchNormalization(momentum=0.9)(fe_A)
    fe_A = LeakyReLU(alpha=0.2)(fe_A)
    fe_A = Dropout(0.15)(fe_A)

    fe_A = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe_A)
    fe_A = BatchNormalization(momentum=0.9)(fe_A)
    fe_A = LeakyReLU(alpha=0.2)(fe_A)
    fe_A = Dropout(0.15)(fe_A)

    fe_A = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe_A)
    fe_A = BatchNormalization(momentum=0.9)(fe_A)
    fe_A = LeakyReLU(alpha=0.2)(fe_A)
    fe_A = Dropout(0.15)(fe_A)

    fe_A = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe_A)
    fe_A = BatchNormalization(momentum=0.9)(fe_A)
    fe_A = LeakyReLU(alpha=0.2)(fe_A)
    fe_A = Dropout(0.15)(fe_A)

    fe_A = Flatten()(fe_A)

    fe_A = Model(inputs=in_image_A, outputs=fe_A)

    # the second branch operates on the second input
    fe_B = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(in_image_B)
    fe_B = BatchNormalization(momentum=0.9)(fe_B)
    fe_B = LeakyReLU(alpha=0.2)(fe_B)
    fe_B = Dropout(0.15)(fe_B)

    fe_B = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe_B)
    fe_B = BatchNormalization(momentum=0.9)(fe_B)
    fe_B = LeakyReLU(alpha=0.2)(fe_B)
    fe_B = Dropout(0.15)(fe_B)

    fe_B = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe_B)
    fe_B = BatchNormalization(momentum=0.9)(fe_B)
    fe_B = LeakyReLU(alpha=0.2)(fe_B)
    fe_B = Dropout(0.15)(fe_B)

    fe_B = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe_B)
    fe_B = BatchNormalization(momentum=0.9)(fe_B)
    fe_B = LeakyReLU(alpha=0.2)(fe_B)
    fe_B = Dropout(0.15)(fe_B)

    fe_B = Flatten()(fe_B)

    fe_B = Model(inputs=in_image_B, outputs=fe_B)

    # the third branch operates on the third input
    fe_C = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(in_image_C)
    fe_C = BatchNormalization(momentum=0.9)(fe_C)
    fe_C = LeakyReLU(alpha=0.2)(fe_C)
    fe_C = Dropout(0.15)(fe_C)

    fe_C = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe_C)
    fe_C = BatchNormalization(momentum=0.9)(fe_C)
    fe_C = LeakyReLU(alpha=0.2)(fe_C)
    fe_C = Dropout(0.15)(fe_C)

    fe_C = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe_C)
    fe_C = BatchNormalization(momentum=0.9)(fe_C)
    fe_C = LeakyReLU(alpha=0.2)(fe_C)
    fe_C = Dropout(0.15)(fe_C)

    fe_C = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe_C)
    fe_C = BatchNormalization(momentum=0.9)(fe_C)
    fe_C = LeakyReLU(alpha=0.2)(fe_C)
    fe_C = Dropout(0.15)(fe_C)

    fe_C = Flatten()(fe_C)

    fe_C = Model(inputs=in_image_C, outputs=fe_C)

    # combine the features from three inputs
    combined = concatenate([fe_A.output, fe_B.output, fe_C.output])

    c_out_layer = Dense(n_classes, activation="softmax", kernel_regularizer=regularizers.l2(0.01))(combined)
    c_model = Model(inputs=[fe_A.input, fe_B.input, fe_C.input], outputs=c_out_layer)
    c_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

    c_model.summary()
    plot_model(c_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    return c_model