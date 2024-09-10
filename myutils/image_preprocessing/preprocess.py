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


### Some methods to try and detect colour cards, but they are failing

def find_color_card(image):
    # load the ArUCo dictionary, grab the ArUCo parameters, and
    # detect the markers in the input image
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image,
                                                       arucoDict, parameters=arucoParams)


def detect_colorchecker(image_path):
    # https://docs.opencv.org/3.1.0/d4/dc6/tutorial_py_template_matching.html
    # Load the image
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    img_rgb = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('inputs/card.png', 0)
    w, h = template.shape[::-1]
    method = cv2.TM_SQDIFF_NORMED
    res = cv2.matchTemplate(img_gray, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img_rgb,top_left, bottom_right, 255, 2)
    cv2.imwrite(os.path.join('example_outputs', 'card.jpg'), img_rgb)

def detect_color_checker(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area, descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for i, contour in enumerate(contours[:40]):  # Check only the 10 largest contours
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the polygon has 4 vertices (rectangular)
        if len(approx) == 4:
            # Check aspect ratio (4:6 = 0.667)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            print(f"Contour {i}: vertices: {len(approx)}, aspect ratio: {aspect_ratio:.2f}")

            if 0.6 < aspect_ratio < 0.75:  # Allow some tolerance
                # Draw the contour on the original image
                cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
                return image, approx

    return image, None

def colour_calibration():
    from plantcv import plantcv as pcv
    rgb_img, path, filename = pcv.readimage('inputs/calib_card.png')
    params = {
        'rgb_img': [rgb_img],
        'adaptive_method': [0,1],
        'block_size': [51, 101, 151],
        'radius': [5,10,20,25,30],
        'min_size': [200,300,400,500,600,700,900, 1000, 1100, 1200, 1300, 1400, 1500]
        # 'threshold_type': ['normal', 'otsu', 'adaptguass'],
        # 'threshvalue': [100, 105, 110, 115, 125, 130, 135, 140, 145, 150],
        # 'blurry': [True, False],
        # 'background': ['light', 'dark']
    }
    import itertools
    keys = list(params)
    i=0
    for values in itertools.product(*map(params.get, keys)):
        i+=1
        try:
            labeled_mask = pcv.transform.detect_color_card(**dict(zip(keys, values)))
            print(values)
            cv2.imwrite(os.path.join('example_outputs', f'labeled_mask_{i}.jpg'), labeled_mask)
        except RuntimeError:
            continue

    # # Use these outputs to create a labeled color card mask
    # mask = pcv.transform.create_color_card_mask(rgb_img='inputs/calib_card.png', radius=10, start_coord=start, spacing=space, ncols=6,
    #                                             nrows=4)


def playing():
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
    ihc_hed = image_rgb  # rgb2hed(image_rgb,channel_axis=-1)

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


if __name__ == '__main__':
    detect_colorchecker('inputs/example_with_card.png')

    # if color_checker_contour is not None:
    #     print("ColorChecker detected!")
    #     cv2.imshow('Detected ColorChecker', result_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # else:
    #     print("ColorChecker not detected.")
    # detect_colorchecker('inputs/calib_card.png')
