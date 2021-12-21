""" Created by Ruben Teunisse 2021 """

import matplotlib.pyplot as plt
import numpy as np

def tune_threshold():
    """ In case of new data, use this function to find the correct threshold value """

    thresholds = [1.5, 2.0, 2.5, 3, 3.5, 4.0]

    plt.figure()
    f, axarr = plt.subplots(6, 6, figsize=(15,15))

    col = 0
    for th in thresholds:
        data = get_CERMEP(path="D:\\Human Skulls\\CERMEP-IDB-MRXFDG",
                          nr_subjects=nr_subjects,
                          threshold=th,
                          trim=trim,
                          ds_factor=2)
        x_train, x_test, y_train, y_test, input_shape = data

        axarr[0, col].title.set_text(f"{th}")
        axarr[0, col].imshow(x_test[80, :, :, 0])
        axarr[1, col].imshow(x_test[130, :, :, 0])
        axarr[2, col].imshow(x_test[155, :, :, 0])
        axarr[3, col].imshow(x_test[170, :, :, 0])
        axarr[4, col].imshow(x_test[190, :, :, 0])
        axarr[5, col].imshow(x_test[210, :, :, 0])


        col += 1
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plt.tight_layout()
    plt.show()

def show_results(model, encoder, x_train, x_test, trim, img_rows, img_cols):
    x_train_encoded = encoder.predict(x_train)
    print("Feature limits")
    print(np.min(x_train_encoded, axis=0),
          np.max(x_train_encoded, axis=0))  # Necessary for knowing the realistic values

    train_sample_x = x_train[155 - trim:191 - trim:35]
    train_sample_y = model.predict(train_sample_x)
    sample_x = x_test[155 - trim:191 - trim:35]
    sample_y = model.predict(sample_x)

    # Plot results
    plt.figure()
    f, axarr = plt.subplots(2, 4)
    axarr[0, 0].imshow(sample_x[0, :, :, 0])
    axarr[0, 0].title.set_text("train input")
    axarr[0, 1].imshow(sample_y[0].reshape(img_rows, img_cols))
    axarr[0, 1].title.set_text("train output")
    axarr[0, 2].imshow(train_sample_x[0, :, :, 0])
    axarr[0, 2].title.set_text("test input")
    axarr[0, 3].imshow(train_sample_y[0].reshape(img_rows, img_cols))
    axarr[0, 3].title.set_text("test output")
    axarr[1, 0].imshow(sample_x[1, :, :, 0])
    axarr[1, 1].imshow(sample_y[1].reshape(img_rows, img_cols))
    axarr[1, 2].imshow(train_sample_x[1, :, :, 0])
    axarr[1, 3].imshow(train_sample_y[1].reshape(img_rows, img_cols))
    plt.show()
