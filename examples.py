import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import App.OctLayers as oct
import matplotlib.colors as colors
from gray2color import gray2color
import cv2
from skimage.color import gray2rgb
import os
import nibabel as nib

def layerMaskExample(obj, combined_str, folder_output):
    num_layers = 10  
    myoct = np.copy(obj.octdata)
    colors = [100,150, 200,300,400,500,600,800,1000,1200]
    composite_image = np.zeros_like(myoct) #, dtype=np.uint8)
    for j in range(128):
        for i in range(num_layers):
            layer_mask = obj.getOctLayerMask(i, i + 1, True)  # Create mask for layers i to i+1
            color = colors[i]
            composite_image[layer_mask] = color

    output_path = os.path.join(folder_output, f'{combined_str}.nii')
    nii_img = nib.Nifti1Image(composite_image.astype(np.uint16), affine=np.eye(4))  # Create NIfTI image with all layers
    nib.save(nii_img, output_path)
    
def etdrsThicknessExample(obj):
    """Demonstrates how to get layer thickness values for the 9 etdrs regions
    Note surfaces are numbered from 0 not 1"""
    thickness1 = obj.getEtdrsThickness(0,3,True)
    thickness2 = obj.getEtdrsThickness(3,5,True)
    thickness = pd.concat([thickness1, thickness2], axis=1)
    
    #TODO: make this a proper test
    # layers 0-3 should be thicker than 3-5 (except at the fovea)


def etdrsIntensityExample(obj):
    """Demonstrates how to get layer intensity values for the 9 etdrs regions
    Note surfaces are numbered from 0 not 1"""
    intensities1 = obj.getEtdrsIntensity(5,6,True)
    intensities2 = obj.getEtdrsIntensity(7,10,True)
    intensities = pd.concat([intensities1, intensities2], axis=1)
    
    #TODO: make this a proper test
    # layers 5 -6 should be dimmer than 7 - 10

if __name__=='__main__':   
    df = pd.read_csv('/cmaldonado/segmentation/nnunet_test.bulk', sep=" ", header=None)    
    for i, (id_col, value_col) in enumerate(zip(df[0], df[1])):
        combined_str = f'{id_col}_{value_col}'
        os.mkdir(f'/cmaldonado/segmentation/Data/NIfTI/nnunet_test/{combined_str}')
        folder_output = f'/cmaldonado/segmentation/Data/NIfTI/nnunet_test/{combined_str}'
        folder_path = f'/cmaldonado/segmentation/Data/IOWA/nnunet_test/fds_test_nnunet/{combined_str}'
        fname_layers = os.path.join(folder_path, f'{combined_str}_Surfaces_Iowa.xml')
        fname_centers = os.path.join(folder_path, f'{combined_str}_GridCenter_Iowa.xml')
        fname = os.path.join(folder_path, f'{combined_str}_OCT_Iowa.fds')

        data_oct = oct.OctLayers(filename=fname_layers,
                             center_filename=fname_centers,
                             raw_filename=fname)
        data_oct.findFovea()
        data_oct.centerData()

        layerMaskExample(data_oct, combined_str, folder_output)
        etdrsThicknessExample(data_oct)
        etdrsIntensityExample(data_oct)
