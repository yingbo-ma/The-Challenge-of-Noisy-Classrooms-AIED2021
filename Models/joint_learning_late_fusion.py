from support_functions import read_excel, spectrogram_data_prepare, optical_flow_data_prepare, define_model
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

label_list = read_excel(label_path)
print("Reading spectrogram data...")
spectrogram_data_0_train, spectrogram_data_1_train, spectrogram_data_2_train, spectrogram_train_data, spectrogram_test_data, test_label, shuffled_list_0, shuffled_list_1, shuffled_list_2 = spectrogram_data_prepare(label_list, spectrogram_image_path, PIXEL, SPECTRO_IMAGE_CHANNELS, SPLIT_RATIO)
print("Reading left face motion data...")
left_optical_flow_data_0_train, left_optical_flow_data_1_train, left_optical_flow_data_2_train, left_optical_flow_train_data, left_optical_flow_test_data = optical_flow_data_prepare(label_list, left_face_optical_flow_image_path, PIXEL, OPTIC_IMAGE_CHANNELS, SPLIT_RATIO, shuffled_list_0, shuffled_list_1, shuffled_list_2)
print("Reading right face motion data...")
right_optical_flow_data_0_train, right_optical_flow_data_1_train, right_optical_flow_data_2_train, right_optical_flow_train_data, right_optical_flow_test_data = optical_flow_data_prepare(label_list, right_face_optical_flow_image_path, PIXEL, OPTIC_IMAGE_CHANNELS, SPLIT_RATIO, shuffled_list_0, shuffled_list_1, shuffled_list_2)

print("Building acoustic model...")
acoustic_model = define_model(input_shape = (PIXEL, PIXEL, SPECTRO_IMAGE_CHANNELS), n_classes = CLASS_NUM)
print("Building left face motion model...")
left_motion_model = define_model(input_shape = (PIXEL, PIXEL, OPTIC_IMAGE_CHANNELS), n_classes = CLASS_NUM)
print("Building right face motion model...")
right_motion_model = define_model(input_shape = (PIXEL, PIXEL, OPTIC_IMAGE_CHANNELS), n_classes = CLASS_NUM)

print("Start training...")

BATCH_SIZE = 128
n_samples = int(BATCH_SIZE / CLASS_NUM)
BATCH_NUM = int(len(spectrogram_train_data) / BATCH_SIZE) + 1
ITERATIONS = 5000

epoch = 0

for i in range(ITERATIONS):

    ix = np.random.randint(0, len(spectrogram_data_0_train), n_samples)

    # prepare training data_0 for acoustic model
    Acoustic_X_supervised_samples_class_0 = np.asarray(spectrogram_data_0_train[ix])
    Acoustic_Y_supervised_samples_class_0 = np.zeros((n_samples, 1))

    # prepare training data_0 for left_motion model
    left_Motion_X_supervised_samples_class_0 = np.asarray(left_optical_flow_data_0_train[ix])
    left_Motion_Y_supervised_samples_class_0 = np.zeros((n_samples, 1))

    # prepare training data_0 for right_motion model
    right_Motion_X_supervised_samples_class_0 = np.asarray(right_optical_flow_data_0_train[ix])
    right_Motion_Y_supervised_samples_class_0 = np.zeros((n_samples, 1))

    ix = np.random.randint(0, len(spectrogram_data_1_train), n_samples)

    # prepare training data_1 for acoustic model
    Acoustic_X_supervised_samples_class_1 = np.asarray(spectrogram_data_1_train[ix])
    Acoustic_Y_supervised_samples_class_1 = np.ones((n_samples, 1))

    # prepare training data_1 for left_motion model
    left_Motion_X_supervised_samples_class_1 = np.asarray(left_optical_flow_data_1_train[ix])
    left_Motion_Y_supervised_samples_class_1 = np.ones((n_samples, 1))

    # prepare training data_1 for right_motion model
    right_Motion_X_supervised_samples_class_1 = np.asarray(right_optical_flow_data_1_train[ix])
    right_Motion_Y_supervised_samples_class_1 = np.ones((n_samples, 1))

    ix = np.random.randint(0, len(spectrogram_data_2_train), n_samples)

    # prepare training data_2 for acoustic model
    Acoustic_X_supervised_samples_class_2 = np.asarray(spectrogram_data_2_train[ix])
    Acoustic_Y_supervised_samples_class_2 = 2 * np.ones((n_samples, 1))

    # prepare training data_2 for left_motion model
    left_Motion_X_supervised_samples_class_2 = np.asarray(left_optical_flow_data_2_train[ix])
    left_Motion_Y_supervised_samples_class_2 = 2 * np.ones((n_samples, 1))

    # prepare training data_2 for right_motion model
    right_Motion_X_supervised_samples_class_2 = np.asarray(right_optical_flow_data_2_train[ix])
    right_Motion_Y_supervised_samples_class_2 = 2 * np.ones((n_samples, 1))

    # concatenate acoustic training data

    Acoustic_Xsup_real = np.concatenate(
        (Acoustic_X_supervised_samples_class_0, Acoustic_X_supervised_samples_class_1, Acoustic_X_supervised_samples_class_2), axis=0)
    Acoustic_Y_sup_real = np.concatenate(
        (Acoustic_Y_supervised_samples_class_0, Acoustic_Y_supervised_samples_class_1, Acoustic_Y_supervised_samples_class_2), axis=0)

    # concatenate left_motion training data

    left_Motion_Xsup_real = np.concatenate(
        (left_Motion_X_supervised_samples_class_0, left_Motion_X_supervised_samples_class_1, left_Motion_X_supervised_samples_class_2), axis=0)
    left_Motion_Y_sup_real = np.concatenate(
        (left_Motion_Y_supervised_samples_class_0, left_Motion_Y_supervised_samples_class_1, left_Motion_Y_supervised_samples_class_2), axis=0)

    # concatenate right_motion training data

    right_Motion_Xsup_real = np.concatenate(
        (right_Motion_X_supervised_samples_class_0, right_Motion_X_supervised_samples_class_1, right_Motion_X_supervised_samples_class_2), axis=0)
    right_Motion_Y_sup_real = np.concatenate(
        (right_Motion_Y_supervised_samples_class_0, right_Motion_Y_supervised_samples_class_1, right_Motion_Y_supervised_samples_class_2), axis=0)

    # update acoustic model
    c_loss, c_acc = acoustic_model.train_on_batch(Acoustic_Xsup_real, Acoustic_Y_sup_real)

    # update left_motion model
    l_m_loss, l_m_acc = left_motion_model.train_on_batch(left_Motion_Xsup_real, left_Motion_Y_sup_real)

    # update left_motion model
    r_m_loss, r_m_acc = right_motion_model.train_on_batch(right_Motion_Xsup_real, right_Motion_Y_sup_real)

    if (i + 1) % (BATCH_NUM * 1) == 0:
        epoch += 1
        print(f"Epoch {epoch}, acoustic model accuracy on training data: {c_acc}")
        print(f"Epoch {epoch}, left_motion model accuracy on training data: {l_m_acc}")
        print(f"Epoch {epoch}, right_motion model accuracy on training data: {r_m_acc}")

        acoustic_y_pred = acoustic_model.predict(spectrogram_test_data, batch_size=60, verbose=0)
        left_motion_y_pred = left_motion_model.predict(left_optical_flow_test_data, batch_size=60, verbose=0)
        right_motion_y_pred = right_motion_model.predict(right_optical_flow_test_data, batch_size=60, verbose=0)
        
        aver_y_pred = np.empty(shape=(len(spectrogram_test_data), CLASS_NUM), dtype='object')
        
        for i in range(len(spectrogram_test_data)):
            for j in range(CLASS_NUM):
                # aver_y_pred[i][j] = ( acoustic_y_pred[i][j] + left_motion_y_pred[i][j] + right_motion_y_pred[i][j] ) / 3
                aver_y_pred[i][j] = 0.4*acoustic_y_pred[i][j] + 0.4*left_motion_y_pred[i][j] + 0.2*right_motion_y_pred[i][j]

        y_pred_bool = np.argmax(aver_y_pred, axis=1)
        print(classification_report(test_label, y_pred_bool))