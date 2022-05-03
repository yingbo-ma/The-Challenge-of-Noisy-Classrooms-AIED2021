from support_functions import read_excel, spectrogram_data_prepare, optical_flow_data_prepare, multiple_inputs_model
import numpy as np
from sklearn.metrics import classification_report

label_path = r"E:\Research Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\new_label.xlsx"
spectrogram_image_path = r"E:\Research Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\Image_Data"
left_face_optical_flow_image_path = r"E:\Research Code\Optical-Flow-Analysis\face_blob_detection\left_faces_optical_flow"
right_face_optical_flow_image_path = r"E:\Research Code\Optical-Flow-Analysis\face_blob_detection\right_faces_optical_flow"

PIXEL = 64
SPECTRO_IMAGE_CHANNELS = 3
OPTIC_IMAGE_CHANNELS = 1
SPLIT_RATIO = 0.75
CLASS_NUM = 3
OPTIC_CLASS_NUM = 1

# Reading Data and Labels
label_list = read_excel(label_path)

spectrogram_train_data, train_label, spectrogram_test_data, test_label, shuffled_list_0, shuffled_list_1, shuffled_list_2 = spectrogram_data_prepare(
    label_list, spectrogram_image_path, PIXEL, SPECTRO_IMAGE_CHANNELS, SPLIT_RATIO)

left_optical_flow_train_data, left_optical_flow_test_data = optical_flow_data_prepare(
    label_list, left_face_optical_flow_image_path, PIXEL, OPTIC_IMAGE_CHANNELS, SPLIT_RATIO, shuffled_list_0,
    shuffled_list_1, shuffled_list_2)

right_optical_flow_train_data, right_optical_flow_test_data = optical_flow_data_prepare(
    label_list, right_face_optical_flow_image_path, PIXEL, OPTIC_IMAGE_CHANNELS, SPLIT_RATIO, shuffled_list_0,
    shuffled_list_1, shuffled_list_2)

# Building Models

c_model = multiple_inputs_model(input_shape_A=(PIXEL, PIXEL, SPECTRO_IMAGE_CHANNELS),
                                input_shape_B=(PIXEL, PIXEL, OPTIC_IMAGE_CHANNELS),
                                input_shape_C=(PIXEL, PIXEL, OPTIC_IMAGE_CHANNELS),
                                n_classes=CLASS_NUM)

# Start Training

BATCH_SIZE = 64
EPOCH = 50

c_model.fit(
    [spectrogram_train_data, left_optical_flow_train_data, right_optical_flow_train_data],
    train_label,
    batch_size = BATCH_SIZE,
    epochs = EPOCH,
    shuffle = True,
    validation_data = ([spectrogram_test_data, left_optical_flow_test_data, right_optical_flow_test_data], test_label)
)

# Start Testing

_, test_acc = c_model.evaluate(
[spectrogram_test_data, left_optical_flow_test_data, right_optical_flow_test_data],
    test_label
)
print('Accuracy: %.2f' % (test_acc * 100))

y_pred = c_model.predict(
    [spectrogram_test_data, left_optical_flow_test_data, right_optical_flow_test_data],
    verbose=0
)
y_pred_bool = np.argmax(y_pred, axis=1)
test_label_bool = np.argmax(test_label, axis=1)
print(classification_report(test_label_bool, y_pred_bool))