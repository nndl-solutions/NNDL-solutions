import network3
import numpy as np
import scipy
import matplotlib.pyplot as plt
import PIL

training_data, _, _ = network3.load_data_shared()
training_x, training_y = training_data
x = training_x.get_value()[0]  # vector corresponding to a 5
x_img = np.reshape(x, (-1, 28))  # recognizable 5
y = training_y.eval()[0]  # label: 5


#### Translate
# Randomly translate the image by -1, 0 or 1 pixel right and down
x_tr = np.random.randint(-1, 2)  # x > 0 means translated to the right
print("translation: {} pixel right".format(x_tr))
y_tr = np.random.randint(-1, 2)  # y > 0 means translated down
print("translation: {} pixel down".format(y_tr))

if x_tr != 0:
    x_img = np.roll(x_img, x_tr, 1)  # 1: x axis
    if x_tr > 0:
        # The image is to be translated to the right
         x_img[:, 0:x_tr] = np.zeros((28, x_tr))
    else:
        # The image is to be translated to the left
        x_img[:, 28+x_tr:] = np.zeros((28, -x_tr))

if y_tr != 0:
    x_img = np.roll(x_img, y_tr, 0)  # 0: y axis
    if y_tr > 0:
        # The image is to be translated up
        x_img[0:y_tr, :] = np.zeros((y_tr, 28))
    else:
        # The image is to be translated down
        x_img[28+y_tr:, :] = np.zeros((-y_tr, 28))


#### Rotate
theta = np.random.uniform(low=-5, high=5)  # degrees counter-clockwise
print("theta: {} degrees".format(theta))
x_img = scipy.ndimage.interpolation.rotate(x_img, theta,
                                           reshape=False,
                                           prefilter=False)

#### Skew
img = PIL.Image.fromarray(x_img)
# upper pixels will be translated to the right by this amount divided by 2:
delta = 2*np.random.randint(-1, 2)  # -2, 0 or 2
print("skew: {} pixels".format(delta))
if delta != 0:
    m = delta / 28
    new_width = 28 + abs(delta)
    img = img.transform((new_width, 28), PIL.Image.AFFINE,
            (1, m, -abs(delta) if delta > 0 else 0, 0, 1, 0))
    # Only keep the center of the new image, keeping the dimensions 28x28:
    img = img.crop((abs(delta)/2, 0, 28 + abs(delta)/2, 28))
    x_img = np.array(img)

plt.imshow(x_img, cmap="binary")
plt.show()