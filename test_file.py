from osgeo import gdal
import numpy as np

def read_tiff_as_array(tiff_path):

    dataset = gdal.Open(tiff_path, gdal.GA_ReadOnly)
    
    if dataset is None:
        print("Could not open the file:", tiff_path)
        return None
    
    # Get image dimensions
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands_count = dataset.RasterCount
    
    # Initialize an empty array to hold the data
    data = np.zeros((height, width, bands_count), dtype=np.float32)
    

    for i in range(1, bands_count + 1):
        band = dataset.GetRasterBand(i)
        data[:, :, i - 1] = band.ReadAsArray()
    
    dataset = None
    
    return data
