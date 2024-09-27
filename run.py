import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm
import rasterio
from regularize import regularize_segmentations
import random 

def save_tif_coregistered_with_img_id(filename, image, img_id_before, img_id_after, poly, channels=1, factor=1):
    height, width = image.shape[0], image.shape[1]
    geotiff_transform = rasterio.transform.from_bounds(poly[0], poly[1],
                                                       poly[2], poly[3],
                                                       width/factor, height/factor)

    with rasterio.open(filename, 'w', driver='GTiff',
                                height=height/factor, width=width/factor,
                                count=channels, dtype='uint8',
                                crs='+proj=latlong',
                                transform=geotiff_transform) as dst:
   # Write bands
        if channels>1:
         for ch in range(0, image.shape[2]):
           dst.write(image[:,:,ch], ch+1)
        else:
           dst.write(image, 1)

        dst.update_tags(img_id_before='{}'.format(img_id_before))
        dst.update_tags(img_id_after='{}'.format(img_id_after))
        
    dst.close()

    return True


mask_dir = '/home/mariapap/CODE/CLv2_eval/PIPELINE_RESULTS/OUTPUT_incorridor/'
before_dir = '/home/mariapap/DATASETS/cl_coreg_docker/May_images/'
after_dir = '/home/mariapap/DATASETS/cl_coreg_docker/July_images/'
#output = './output/'

ids = os.listdir('/home/mariapap/DATASETS/cl_coreg_docker/May_images/')
random.shuffle(ids)

#os.makedirs(output, exist_ok=True)

for _, id in enumerate(tqdm(ids)):
    # make a temporary folder, three subfolders: rgb, seg, reg_out
    os.makedirs('temp', exist_ok=True)
    os.makedirs('temp/rgb', exist_ok=True)
    os.makedirs('temp/seg', exist_ok=True)
    os.makedirs('temp/reg_out', exist_ok=True)

    before = before_dir + id
    after = after_dir + id
    mask = mask_dir + 'output_{}'.format(id)
    
    # preprocess the mask
    # get the second channel of the mask and save it as a new tif in the seg folder
    mask_img = cv2.imread(mask, cv2.IMREAD_UNCHANGED)
    mask_img = rasterio.open(mask)
    bounds = mask_img.bounds
    img1_id = mask_img.tags()['img_id_before']
    img2_id = mask_img.tags()['img_id_after']
    mask_img = mask_img.read()
    mask_img = np.transpose(mask_img, (1,2,0))
    mask_img = mask_img[:,:,[2,1,0]]
    mask_after = mask_img[:,:,1]
    cv2.imwrite('temp/seg/after.tif', mask_after)

    mask_before = mask_img[:,:,2]
    cv2.imwrite('temp/seg/before.tif', mask_before)

    # copy the before and after images to the rgb folder
    img_before = cv2.imread(before)
    img_after = cv2.imread(after)
    cv2.imwrite('temp/rgb/before.tif', img_before)
    cv2.imwrite('temp/rgb/after.tif', img_after)

    # regularize the segmentation
    regularize_segmentations(img_folder='temp/rgb/*.tif', seg_folder='temp/seg/*.tif', out_folder='temp/reg_out/', in_mode="semantic", out_mode="instance", samples=None)

    # postprocess the output
    # save output 'before' as channel 2 with nonzero values=255, 'after' as channel 1 with nonzero values=255, for channel 0 all 0s
    # save the output as a tif
    output_img = np.zeros_like(mask_img)
    output_img[:,:,0] = 0
    output_img[:,:,1] = cv2.imread('temp/reg_out/after.tif', cv2.IMREAD_UNCHANGED)
    output_img[:,:,2] = cv2.imread('temp/reg_out/before.tif', cv2.IMREAD_UNCHANGED)
    output_img[output_img > 0] = 255

    cv2.imwrite('/home/mariapap/CODE/CLv2_eval/PIPELINE_RESULTS/OUTPUT_reg/' + id, output_img)
    save_tif_coregistered_with_img_id('/home/mariapap/CODE/CLv2_eval/PIPELINE_RESULTS/OUTPUT_reg/' + id, output_img[:,:,[2,1,0]], img1_id, img2_id, bounds, channels=3)

    # clear temp folders

    shutil.rmtree('temp')
