import os

import cv2
import numpy as np
import skimage as ski
from matplotlib import pyplot as plt
from skimage import color, filters, exposure, io, data
from skimage.color import hed2rgb, rgb2hed
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value


def gaussian_blur(image, sigma):
    pass


if __name__ == '__main__':
    img_file = os.path.join('..', 'example_images', 'A0006-20240408_145101.jpg')
    image_rgb = cv2.imread(img_file)
    # image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # display the image

    cv2.imwrite(os.path.join('example_outputs', 'unblurred.jpg'), image_rgb)
    # # Convert the image to grayscale if it's not already
    # if img.ndim == 3 and img.shape[2] == 3:
    #     image = color.rgb2gray(img)

    # Apply Gaussian blur
    # channel_axis indicates the colour channel
    # preserve_range is needed for these formats.
    blurred = filters.gaussian(image_rgb, sigma=10, channel_axis=-1, preserve_range=True)

    cv2.imwrite(os.path.join('example_outputs', 'blurred.jpg'), blurred)

    # Contrast stretching
    p2, p98 = np.percentile(image_rgb, (1, 99))
    img_rescale = exposure.rescale_intensity(image_rgb, in_range=(p2, p98))

    cv2.imwrite(os.path.join('example_outputs', 'constrast_stretched.jpg'), img_rescale)


    # @adapt_rgb(each_channel)
    # def equalize_hist_each(image):
    #     return exposure.equalize_hist(image)
    #
    # # Equalization
    # img_eq = equalize_hist_each(img)
    # cv2.imwrite(os.path.join('example_outputs', 'img_eq.jpg'), img_eq)
    #
    #
    # # Adaptive Equalization
    # img_adapteq = io.imread(img_file)
    # for channel in range(img_adapteq.shape[2]):  # equalizing each channel
    #     channel_values = img_adapteq[:, :, channel]
    #     img_adapteq[:, :, channel] = exposure.equalize_adapthist(channel_values)
    # cv2.imwrite(os.path.join('example_outputs', 'img_adapteq.jpg'), img_adapteq)

    # Separate the stains from the IHC image
    # image_rgb = data.immunohistochemistry()
    ihc_hed = image_rgb#rgb2hed(image_rgb,channel_axis=-1)

    # Create an RGB image for each of the stains
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = np.stack((ihc_hed[:, :, 0], null, null), axis=-1)
    cv2.imwrite(os.path.join('example_outputs', 'b.jpg'), ihc_h)
    ihc_e = np.stack((null, ihc_hed[:, :, 1], null), axis=-1)
    cv2.imwrite(os.path.join('example_outputs', 'g.jpg'), ihc_e)

    ihc_d = np.stack((null, null, ihc_hed[:, :, 2]), axis=-1)
    cv2.imwrite(os.path.join('example_outputs', 'r.jpg'), ihc_d)

    # backSub = cv2.createBackgroundSubtractorMOG2()
    # fgMask = backSub.apply(image_rgb)
    # cv2.imwrite(os.path.join('example_outputs', 'fgMask.jpg'), fgMask)



