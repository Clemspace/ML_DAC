import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from sklearn.linear_model import Lasso

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
    plt.colorbar()



def plot_error(datax,datay,f,step=10):
    grid,x1list,x2list=make_grid(xmin=-4,xmax=4,ymin=-4,ymax=4)
    plt.contourf(x1list,x2list,np.array([f(datax,datay,w) for w in grid]).reshape(x1list.shape),25)
    plt.colorbar()
    plt.show()

def init_USPS_1v1(val1,val2):
    """
    Sépare deux classes au label données en argument dans les données USPS, et retourne les sets et labels correspondants déja étiquetés
    """

    train_datax, train_datay = load_usps("USPS_train.txt")
    test_datax, test_datay = load_usps("USPS_test.txt")


    train_x = train_datax[np.logical_or(train_datay == val1, train_datay == val2), :]

    train_y = train_datay[np.logical_or(train_datay == val1, train_datay == val2)]
    train_y = np.where(train_y == val1, 1, -1)

    test_x = test_datax[np.logical_or(test_datay == val1, test_datay == val2), :]

    test_y = test_datay[np.logical_or(test_datay == val1, test_datay == val2)]
    test_y = np.where(test_y == val1, 1, -1)

    return train_x, train_y, test_x, test_y

def init_USPS_1vAll(isolated_value):
    """
    Sépare deux classes au label données en argument dans les données USPS, et retourne les sets et labels correspondants déja étiquetés
    """


    train_datax, train_datay = load_usps("USPS_train.txt")
    test_datax, test_datay = load_usps("USPS_test.txt")



    train_y = np.where(train_y == valplus, 1, -1)


    test_y = np.where(test_y == isolated_value, 1, -1)

    return train_datax, train_y, test_datax, test_y


def read_im(fn):

    #with open(fn,"r") as f:
    img = plt.imread(fn)[:,:,:3]

    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        img = img/info.max

    im_h, im_l, _ = img.shape

    # Transform into array of pixels (3 dim)
    pixels = img.reshape((im_h , im_l, 3)) # in RGB
    img = 2*img - 1

    pixels_hsv = clr.rgb_to_hsv(pixels) # convert to HSV
    pixels_hsv= 2*pixels_hsv - 1
    return pixels_hsv


def display_im(img):
    """
    displays given image at format RGB
    """
    plt.figure()


    img = (img + 1)/2


    im = clr.hsv_to_rgb(img)

    plt.imshow(im)


def get_patch(i, j, h, img):
    """
    (i, j) : coordonnées du point au centre
    h : longueur du patch
    """
    imin = i - (h//2)
    jmin = j - (h//2)
    return img[imin:imin+h, jmin:jmin+h, :]

def delete_rect(img, i, j, height, width):
    """
    (i, j) : coordonnées du point au centre
    """
    i0 = i - (height//2)
    j0 = j - (width//2)
    img[i0:(i0+height), j0:(j0+width), :] = -100

def patch_to_vect(patch):

    return patch.reshape(-1)

def vect_to_patch(vect):

    h = int(np.sqrt(vect.size//3))

    return vect.reshape(h, h, 3)

def add_noise(img, prc):
    imgcopy = img.copy()
    noise = np.random.random(imgcopy.shape[:2]) <= prc
    for i in range(imgcopy.shape[2]):
        imgcopy[:, :, i] = np.where(noise, -100, imgcopy[:, :, i])
    return imgcopy
