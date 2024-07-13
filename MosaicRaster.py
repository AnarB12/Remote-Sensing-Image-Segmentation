import arcpy
from arcpy.sa import *
import os
from arcpy import env

def CreateMosaicRaster(input_dir, output_dir, gdb_name="OutputData.gdb",
                       mosaic_name="mosaic_output", pixel_type="8_BIT_UNSIGNED",
                       cell_size="", number_of_bands="1"):
    """
    Creates a mosaic raster from TIFF images located in a specified input directory.

    Parameters:
    - input_dir (str): The directory containing TIFF images to be mosaicked.
    - output_dir (str): The directory where the output data (mosaic) will be stored.
    - gdb_name (str, optional): Name for the output geodatabase. Defaults to "OutputData.gdb".
    - mosaic_name (str, optional): Name for the mosaic raster output. Defaults to "mosaic_output".
    - pixel_type (str, optional): Pixel type for the mosaic raster. Defaults to "8_BIT_UNSIGNED".
    - cell_size (str or float, optional): Cell size for the mosaic raster. 
                                          Uses the source data's cell size if not specified.
    - number_of_bands (str, optional): Number of bands for the mosaic raster. Defaults to "1".

    Note:
    This function requires the ArcPy and Spatial Analyst extensions. 
    Ensure they are available and licensed for use before running this function.
    """

    # Initialize progress tracking
    steps = ["Listing TIFF files", "Creating mosaic"]
    total_steps = len(steps)
    current_step = 1

    env.overwriteOutput = True
    env.extent = "MAXOF"
    workspace = input_dir
    outputLocation = os.path.join(output_dir, gdb_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not arcpy.Exists(outputLocation):
        arcpy.management.CreateFileGDB(output_dir, gdb_name)

    print(f"[{current_step}/{total_steps}] {steps[current_step-1]}...")
    current_step += 1
    env.workspace = outputLocation

    rasterList = [os.path.join(workspace, file) for file in os.listdir(workspace) if file.endswith('.TIF')]
    if not rasterList:
        print("No TIFF files found in the specified directory.")
        return

    print(f"[{current_step}/{total_steps}] {steps[current_step-1]}...")
    current_step += 1
    mosaicRasterPath = os.path.join(outputLocation, mosaic_name)

    arcpy.management.MosaicToNewRaster(input_rasters=rasterList,
                                       output_location=outputLocation,
                                       raster_dataset_name_with_extension=mosaic_name,
                                       pixel_type=pixel_type,
                                       cellsize=cell_size,
                                       number_of_bands=number_of_bands,
                                       mosaic_method="MAXIMUM")

    print(f"Mosaic created successfully at: {mosaicRasterPath}")

if __name__ == "__main__":
    
    input_dir = r"F:\DEEP_LEARNING_GIS\Predictions\2023_6in_AOI_LULC_512\Pred_patches"
    output_dir = r"F:\DEEP_LEARNING_GIS\Predictions"
    gdb_name = "LC_Prediction.gdb"
    mosaic_name = "LC_Prediction_2023"

    CreateMosaicRaster(
            input_dir=input_dir,
            output_dir=output_dir,
            gdb_name=gdb_name,
            mosaic_name=mosaic_name
        )
