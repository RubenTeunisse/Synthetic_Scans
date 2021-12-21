""" Created by Ruben Teunisse 2021 """

import joblib
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import math

# Get models
#model = joblib.load('./model')
#encoder = joblib.load('./encoder')
decoder = joblib.load('./decoder_high_res_3')

#min_feat_vals = np.array([-0.97376126, -0.47918788, -0.1379233,  -5.6186004]) # Low_res
#max_feat_vals = np.array([5.6411767,  0.04488187, 0.01205764, 2.259701]) # Low_res
# min_feat_vals = np.array([-1.3467636, -1.4574244, -1.5644351, -1.1122341]) # High res
# max_feat_vals = np.array([1.574167,  1.6921705, 1.7055522, 2.971624 ]) # High res
# min_feat_vals = np.array([-1.6395069, -1.6709113, -1.4172486, -1.7495339]) # High res, th 2.75
# max_feat_vals = np.array( [1.2684978, 1.7310706, 1.7803863, 1.0644412]) # High res, th 2.75
min_feat_vals = np.array([-1.6376034, -2.2450988, -1.309281,  -0.722901]) # model 3
max_feat_vals = np.array( [0.8542986, 1.383035,  1.2422124, 2.4490302]) # model 3

# Get GUI functions
def show():
    # Decode feature values
    feat_vals = np.array([f1_slider.get(), f2_slider.get(), f3_slider.get(), f4_slider.get()])
    print(feat_vals)
    img = decoder.predict(feat_vals[None,None,:])[0,0,:]

    # Reshape
    cols = int(math.sqrt(img.shape[0]))
    rows = cols
    img = img.reshape((rows,cols))
    #img = img[img>th_slider.get()]
    #print(img[10:16, 16])
    print(root.th_on)

    # Threshold
    if root.th_on:
        #avg = np.mean(img)
        #std = np.std(img)
        #img = img > avg + std * th_slider.get()  #(np.max(img) - np.min(img)) * th_slider.get() + np.min(img)
        img = img > (np.max(img) - np.min(img)) * th_slider.get() + np.min(img)

    # Get image
    root.image = plt.imshow(img)

    # Show image
    im.set_data(img)
    canvas.draw()

def slider_move(val):
    show()

def tick_th():
    root.th_on = not root.th_on
    show()

def get_random_feats():
    values = []

    for min, max in zip(min_feat_vals, max_feat_vals):
        value = np.random.uniform(min, max)
        values.append(value)

    return np.array(values)

def gen_imgs():
    imgs = []

    for _ in range(int(root.nr_imgs.get(1.0, 'end'))):
        # Get random feats
        feat_vals = get_random_feats()

        # Predict image
        img = decoder.predict(feat_vals[None,None,:])[0,0,:]

        # Reshape
        cols = int(math.sqrt(img.shape[0]))
        rows = cols
        img = img.reshape((rows, cols))
        imgs.append(img)

    joblib.dump(np.array(imgs), f'./{int(root.nr_imgs.get(1.0, "end"))}_images')


# Get GUI
root = Tk()
root.title('Synthetic Scans Generator')
res1 = (max_feat_vals[0] - min_feat_vals[0]) / 100
res2 = (max_feat_vals[1] - min_feat_vals[1]) / 100
res3 = (max_feat_vals[2] - min_feat_vals[2]) / 100
res4 = (max_feat_vals[3] - min_feat_vals[3]) / 100
f1_slider = Scale(root, from_=min_feat_vals[0], to=max_feat_vals[0], orient=HORIZONTAL, digits=4, resolution=res1, length=300, command=slider_move)
f2_slider = Scale(root, from_=min_feat_vals[1], to=max_feat_vals[1], orient=HORIZONTAL, digits=4, resolution=res2, length=300, command=slider_move)
f3_slider = Scale(root, from_=min_feat_vals[2], to=max_feat_vals[2], orient=HORIZONTAL, digits=4, resolution=res3, length=300, command=slider_move)
f4_slider = Scale(root, from_=min_feat_vals[3], to=max_feat_vals[3], orient=HORIZONTAL, digits=4, resolution=res4, length=300, command=slider_move)
f1_slider.set(min_feat_vals[0])
f2_slider.set(min_feat_vals[1])
f3_slider.set(min_feat_vals[2])
f4_slider.set(min_feat_vals[3])

f1_slider.pack()
f2_slider.pack()
f3_slider.pack()
f4_slider.pack()

# Threshold slider
#th_label = Label(root, text="Threshold")
root.th_on = 0
th_checkbox = Checkbutton(root, text="Threshold", variable=root.th_on, command=tick_th)
th_checkbox.pack()
th_slider = Scale(root, from_=0, to=1, orient=HORIZONTAL, digits=4, resolution=0.01, length=300, command=slider_move)
th_slider.pack()

Button(root, text='Show', command=show).pack()

# Probability/TH map
root.image = plt.imread('min_val_plot.jpg')
fig = plt.figure(figsize=(5,4))
im = plt.imshow(root.image)
ax = plt.gca()
ax.set_xticklabels([])
ax.set_yticklabels([])
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

# Image Generator
root.nr_imgs = Text(
    root,
    height=1,
    width=5
)
root.nr_imgs.pack()
Button(root, text='Generate', command=gen_imgs).pack()

mainloop()
