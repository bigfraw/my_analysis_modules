import cv2
import os
import natsort as ns
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

def save_plt(folder_name, file_name, dpi=200):
    """
    This function saves the current plot as a PNG file in a specified folder.

    Inputs:
    - folder_name: the name of the folder where the file will be saved
    - file_name: the name of the file
    - dpi: the resolution of the image

    Returns:
    - None
    """

    current_path = os.getcwd()
    new_path = os.path.join(current_path, folder_name)
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    plt.savefig('{}\\{}.png'.format(folder_name, file_name), dpi=dpi)
    plt.close()


def import_folder(folder, filetype):
    """
    This function imports all the images in a folder and returns a list of them.

    Inputs:
    - folder: the folder where the images are stored
    - type: the type of the images (e.g. 'png', 'jpg', 'jpeg')

    Returns:
    - imgs_original: a list of the images
    """

    imgs_original = []
    i=1
    while True:
        try:
            img = cv2.imread('{}\out{}.{}'.format(folder,i,filetype), 0)
            img[0]
            imgs_original.append(img)
            i += 1
        except:
            break
    return imgs_original

def create_gif(image_folder, gif_name, duration, filetype):
    """
    This function creates a GIF from a folder of images.

    Inputs:
    - image_folder: the folder where the images are stored
    - gif_name: the name of the GIF file (include .gif)
    - duration: the duration of each frame in the GIF
    - filetype: the type of the images (e.g. 'png', 'jpg', 'jpeg')

    Returns:
    - None
    """

    images = []

    # Iterate through the folder and open each image
    for filename in tqdm(ns.natsorted(os.listdir(image_folder))):
        if filename.endswith(f'.{filetype}'):
            filepath = os.path.join(image_folder, filename)
            images.append(Image.open(filepath))

    # Save the images as a GIF
    images[0].save("{}\\{}".format(image_folder, gif_name), save_all=True,
                   append_images=images[1:], duration=duration)