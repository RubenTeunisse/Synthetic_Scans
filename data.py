""" Created by Ruben Teunisse 2021 """

import nibabel as nib
import numpy as np
import os

def get_1_UCL(path, norm=True, threshold=True, ds_factor=1):
    print("Start reading data")
    nii_img = nib.load(path)
    data = nii_img.get_fdata()

    if norm:
        data = (data - np.mean(data)) / np.std(data)
    if threshold:
        data = data > 1.5
    data = data[::ds_factor, ::ds_factor, ::1]

    data = np.swapaxes(data, 0, 2)
    x_train = data[0::2]
    x_test = data[1::2]

    # Set Shape
    img_rows, img_cols = x_train.shape[1], x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    y_train = x_train.reshape(-1, img_rows * img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    y_test = x_test.reshape(-1, img_rows * img_cols)

    input_shape = (img_rows, img_cols, 1)

    # Normalize images
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return x_train, x_test, y_train, y_test, input_shape


def get_CERMEP(path, nr_subjects=37, norm=True, threshold=1.5, trim=30, ds_factor=1):
    print("Start reading data")

    scans = []

    dirs = os.listdir(path)
    count = 1
    for name in dirs:
        if count > nr_subjects:
            break
        if name[:3] == 'sub':
            count += 1
            print(os.path.join(path, name))
            sub_dir = os.path.join(path, name)
            sub_dir += '/ct/' + name + '_ct.nii.gz'
            nii_img = nib.load(sub_dir)
            data = nii_img.get_fdata()

            if norm:
                data = (data - np.mean(data)) / np.std(data)
            if threshold:
                data = data > threshold
            data = data[::ds_factor, ::ds_factor, ::1]

            data = np.swapaxes(data, 0, 2)
            data = data[trim:-int(trim/4)]

            scans.append(data)

    train_len = int((len(scans)/4) * 3)
    scans = np.array(scans)
    x_train = scans[:train_len]
    x_test = scans[train_len:]

    # Set Shape
    img_rows, img_cols = x_train.shape[2], x_train.shape[3]
    x_train = x_train.reshape((-1,img_rows,img_cols))
    x_test = x_test.reshape((-1,img_rows,img_cols))
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    y_train = x_train.reshape(-1, img_rows * img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    y_test = x_test.reshape(-1, img_rows * img_cols)

    input_shape = (img_rows, img_cols, 1)

    # Normalize images
    x_train = x_train.astype('float32') # Uhhh
    x_test = x_test.astype('float32')



    return x_train, x_test, y_train, y_test, input_shape
