import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm

from regularize import regularize_segmentations
import rasterio
from shapely import geometry

#mask = './images/mask_5996668.tif'
#before = './images/before_5996668.tif'
#after = './images/after_5996668.tif'
#output = './output/'

def save_tif_coregistered(filename, image, poly, channels=1, factor=1):
    height, width = image.shape[0], image.shape[1]
    geotiff_transform = rasterio.transform.from_bounds(poly.bounds[0], poly.bounds[1],
                                                       poly.bounds[2], poly.bounds[3],
                                                       width/factor, height/factor)

    new_dataset = rasterio.open(filename, 'w', driver='GTiff',
                                height=height/factor, width=width/factor,
                                count=channels, dtype='uint8',
                                crs='+proj=latlong',
                                transform=geotiff_transform)

    # Write bands
    if channels>1:
     for ch in range(0, image.shape[2]):
       new_dataset.write(image[:,:,ch], ch+1)
    else:
       new_dataset.write(image, 1)
    new_dataset.close()

    return True


def reg_preds(mask, before, after, id):
#os.makedirs(output, exist_ok=True)

# make a temporary folder, three subfolders: rgb, seg, reg_out
    os.makedirs('temp', exist_ok=True)
    os.makedirs('temp/rgb', exist_ok=True)
    os.makedirs('temp/seg', exist_ok=True)
    os.makedirs('temp/reg_out', exist_ok=True)

# preprocess the mask
# get the second channel of the mask and save it as a new tif in the seg folder
    mask_img = cv2.imread(mask, cv2.IMREAD_UNCHANGED)
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
    output_img[:,:,2] = 0
    output_img[:,:,1] = cv2.imread('temp/reg_out/after.tif', cv2.IMREAD_UNCHANGED)
    output_img[:,:,0] = cv2.imread('temp/reg_out/before.tif', cv2.IMREAD_UNCHANGED)
    output_img[output_img > 0] = 255

    bounds = rasterio.open(mask).bounds
    bounds = geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top)

    save_tif_coregistered('./output/' + '{}.tif'.format(id), output_img, bounds, channels=3)

#    cv2.imwrite('./output/' + '{}.tif'.format(id), output_img)

# clear temp folders

    shutil.rmtree('temp')


pre_ids = os.listdir('/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/PIPELINE_RESULTS_natnew2/OUTPUT_filtered')
ids = []
for pid in pre_ids:
    pf = pid.find('region_')
    ids.append(pid[pf+7:-4])

mask_dir = '/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/PIPELINE_RESULTS_natnew2/OUTPUT_filtered/'
before_dir = '/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/PIPELINE_RESULTS_natnew2/BEFORE_REGISTERED/'
after_dir = '/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/NatBasemaps/basemaps_natfuel_test/images/B/'

for _, id in enumerate(tqdm(ids)):
  mask = mask_dir + 'output_data_region_{}.tif'.format(id)
  before = before_dir + 'before_transformed_data_region_{}.tif'.format(id)
  after = after_dir + 'data_region_{}.tif'.format(id)

  reg_preds(mask, before, after, id)


