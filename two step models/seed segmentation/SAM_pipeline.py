import json
import os
from time import process_time

import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def SAM_instance(infile: str, outdir: str):
    """
    :param infile: Path to the input image file.
    :param outdir: Path to the directory where the annotated images will be saved.
    :return: A tuple containing the number of seeds, a list of non-background masks, and the shape of the input image.
    """
    import torch
    import torchvision
    torch.cuda.empty_cache()  # empty the cache to not blow up memory. It helps but doesn't fully resolve issue
    import gc
    # del variables
    gc.collect()

    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    # print(torch.cuda.memory_summary())  # get a memort summary

    from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

    # Specify what segmentation models need to output in order to evaluate and pass onto classifier
    # - Number of seeds
    # - Seperate seed images
    # - Masks for evaluation
    # - Time taken

    # from https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
    image = cv2.imread(infile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # plt.axis('on')
    # plt.show()

    t = process_time()

    # all these versions too big for my machine
    _data_path = os.environ['KEWDATAPATH']

    # with !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    # sam_checkpoint = os.path.join(_data_path, 'model_checkpoints', 'SAM', "sam_vit_h_4b8939.pth")
    # sam_checkpoint = os.path.join(_data_path, 'model_checkpoints', 'SAM', "sam_vit_l_0b3195.pth")
    sam_checkpoint = os.path.join(_data_path, 'model_checkpoints', 'SAM', "sam_vit_b_01ec64.pth")
    # model_type = "vit_h"
    model_type = "vit_b"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    # Play with these hyperparams
    model = SamAutomaticMaskGenerator(sam, points_per_batch=16,
                                      # points_per_batch (int): Sets the number of points run simultaneously # by the model. Higher numbers may be faster but use more GPU memory.
                                      pred_iou_thresh=0.86,
                                      stability_score_thresh=0.92,
                                      crop_n_points_downscale_factor=2,
                                      min_mask_region_area=100,  # Requires open-cv to run post-processing
                                      )
    masks = model.generate(image)

    # choose the biggest area
    # could maybe improve this by excluding anything with area>some value
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    background_mask = sorted_masks[0]['segmentation']

    non_background_masks = sorted_masks[1:]

    plt.figure(figsize=(20, 20))
    plt.imshow(image)

    show_anns(non_background_masks)
    plt.axis('off')
    if not os.path.exists(os.path.join(outdir, 'annotated_images')):
        os.mkdir(os.path.join(outdir, 'annotated_images'))
    plt.savefig(os.path.join(outdir, 'annotated_images', f"{os.path.basename(file_image)}_SAM.png"))
    plt.close()


    output_annotations = non_background_masks
    for m in output_annotations:
        m['category_id'] = 1 # Set category to 1 i.e. a seed
        # Convert the binary mask to polygons to save in coco
        m['segmentation'] = [get_polygons_from_SAM_mask(m)]

    number_seeds = len(non_background_masks)
    print(f'Number of seeds: {number_seeds}')
    elapsed_time = process_time() - t
    print(f'Process time: {elapsed_time}')

    return number_seeds, output_annotations, image.shape

def get_polygons_from_SAM_mask(ann):
    """
    :param ann: Annotation dictionary containing segmentation information
    :return: List of flattened polygons representing the annotation

    """
    # From https://github.com/facebookresearch/segment-anything/issues/121
    # TODO: needs flattening? Currently messy and need to check if works
    from pycocotools import mask as mask_utils
    m = ann['segmentation']
    if isinstance(m, np.ndarray) and m.dtype == bool:
        m = mask_utils.encode(np.asfortranarray(m))
    elif isinstance(m, dict) and 'counts' in m and 'size' in m:
        pass  # Already in RLE format
    else:
        print("Invalid segmentation format:", m)
        raise ValueError

    mask = mask_utils.decode(m)
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [contour.reshape(-1, 2).tolist() for contour in contours]
    flat_list = []

    for xs in polygons[0]:
        for x in xs:
            flat_list.append(x)
    # contours = [np.squeeze(contour) for contour in contours]  # Convert contours to the correct shape
    # contours = [np.atleast_2d(contour) for contour in contours]
    return flat_list

if __name__ == '__main__':

    if not os.path.exists(os.path.join('example_outputs', 'coco_outputs')):
        os.mkdir(os.path.join('example_outputs', 'coco_outputs'))

    # This builds a Coco output with labels 'Seed'
    out_dict = {'licences': [{'id': 0, 'name': '', 'url': ''}],
                'info': {'contributor': 'SAM', 'date_created': '', 'description': '', 'url': '', 'version': '', 'year': ''},
                'categories': [{'id': 1, 'name': 'Seed'}], 'images': [], 'annotations': []}
    img_id = 1
    ann_id = 1
    for file_image in os.listdir(os.path.join(_repo_path, 'SeedSegmentation', 'data', 'annotations', 'pablo_examples', 'images')):
        n_seeds, annotations, image_shape = SAM_instance(
            os.path.join(_repo_path, 'SeedSegmentation', 'data', 'annotations', 'pablo_examples', 'images', file_image),
            'example_outputs')
        out_dict['images'].append({'file_name': file_image, 'height': image_shape[0], 'width': image_shape[1], 'id': img_id})

        for m in annotations:
            m['image_id'] = img_id
            m['id'] = ann_id
            out_dict['annotations'].append(m)
            ann_id += 1

        img_id += 1

        with open(os.path.join('example_outputs', 'coco_outputs', 'result.json'), 'w') as fp:
            json.dump(out_dict, fp)
    # if not os.path.exists(os.path.join('example_outputs', 'summary')):
    #     os.mkdir(os.path.join('example_outputs', 'summary'))
