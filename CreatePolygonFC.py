import arcpy
from arcpy.sa import *
import os
from arcpy import env


def CreatePolygonPredictions(input_dir, output_dir, gdb_name="OutputData.gdb",
                             mosaic_name="mosaic_output", reclassified_name="None",
                             polygon_name="mosaic_polygon", pixel_type="8_BIT_UNSIGNED",
                             cell_size="", number_of_bands="1", reclass_map=None,
                             simplify_option="SIMPLIFY", create_multipart="SINGLE_OUTER_PART",
                             max_vertices_per_feature=None):
    """
    Creates a polygon feature class from TIFF images located in a specified input directory by 
    performing  mosaicking, reclassification, and conversion from raster to 
    polygon feature class. This function also handles the creation of necessary 
    directories and geodatabase, and ensures clean-up of intermediate data.


    Parameters:
    - input_dir (str): The directory containing TIFF images to be mosaicked.
    - output_dir (str): The directory where the output data (mosaic, reclassified raster, 
                        and polygon) will be stored.
    - gdb_name (str, optional): Name for the output geodatabase. Defaults to "OutputData.gdb".
    - mosaic_name (str, optional): Name for the mosaic raster output. Defaults to "mosaic_output".
    - reclassified_name (str, optional): Name for the reclassified raster output. 
                                        Defaults to "reclassified_mosaic".
    - polygon_name (str, optional): Name for the output polygon feature class. 
                                    Defaults to "mosaic_polygon".
    - pixel_type (str, optional): Pixel type for the mosaic raster. Defaults to "8_BIT_UNSIGNED".
    - cell_size (str or float, optional): Cell size for the mosaic raster. Uses the source data's 
                                          cell size if not specified.
    - number_of_bands (str, optional): Number of bands for the mosaic raster. Defaults to "1".
    - reclass_map (list of lists, optional): Defines the reclassification rules as a list of 
                                             [fromValue, toValue] pairs.
      Defaults to None, which applies a predefined map.
    - simplify_option (str, optional): Option for simplifying polygons. Can be "SIMPLIFY" or 
                                       "NO_SIMPLIFY". Defaults to "SIMPLIFY".
    - create_multipart (str, optional): Option to create multipart features. Can be "SINGLE_OUTER_PART" 
                                        or "MULTIPLE_OUTER_PART". Defaults to "SINGLE_OUTER_PART".
    - max_vertices_per_feature (int, optional): Maximum number of vertices per feature. 
                                                No limit if not specified.


    Note:
    This function requires the ArcPy and Spatial Analyst extensions. 
    Make sure they are available and licensed for use before running this function.
    """


    # # 
    # arcpy.CheckExtension("Spatial")



    steps = ["Listing TIFF files", "Creating mosaic",
             "Reclassifying raster", "Converting to polygon feature class",
             "Cleaning up intermediate data"]
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
   
    print("Mosaic created successfully.")


    print(f"[{current_step}/{total_steps}] {steps[current_step-1]}...")
    current_step += 1


    if reclass_map is None:
        reclass_map = RemapValue([[0, "NODATA"], [1, 1]])
    else:
        reclass_map = RemapValue(reclass_map)


    reclassifiedRasterPath = os.path.join(outputLocation, reclassified_name)


    reclassifiedRaster = Reclassify(mosaicRasterPath,
                                    "Value",
                                    reclass_map,
                                    missing_values="NODATA")
   
    reclassifiedRaster.save(reclassifiedRasterPath)
    print("Reclassified raster created successfully.")


    print(f"[{current_step}/{total_steps}] {steps[current_step-1]}...")
    current_step += 1
    polygonFeatureClassPath = os.path.join(outputLocation, polygon_name)


    arcpy.conversion.RasterToPolygon(reclassifiedRaster,
                                     polygonFeatureClassPath,
                                     simplify_option,
                                     "Value",
                                     create_multipart,
                                     max_vertices_per_feature)
   
    print(f"Polygon feature class created at: {polygonFeatureClassPath}")


    print(f"[{current_step}/{total_steps}] {steps[current_step-1]}...")
    arcpy.management.Delete(mosaicRasterPath)
    print(f"Mosaic raster deleted: {mosaicRasterPath}")
    # arcpy.management.Delete(reclassifiedRasterPath)
    # print(f"Reclassified raster deleted: {mosaicRasterPath}")


    # # Check in the Spatial Analyst extension
    # arcpy.CheckInExtension("Spatial")


if __name__ == "__main__":
    
    input_dir = r"F:\DEEP_LEARNING_GIS\Predictions\2023_3in_imgPatches_1024\Pred_patches"
    output_dir = r"F:\DEEP_LEARNING_GIS\Predictions"
    gdb_name = "ImperviousSurface.gdb"
    mosaic_name = "mosaic_output"
    reclassified_name = "Impervious_raster_2023"
    polygon_name = "Impervious_polygon_2023"
    reclass_map = [[0, "NODATA"], [1, 1]]

    CreatePolygonPredictions(
            input_dir=input_dir,
            output_dir=output_dir,
            gdb_name=gdb_name,
            mosaic_name=mosaic_name,
            reclassified_name=reclassified_name,
            polygon_name=polygon_name,
            reclass_map=reclass_map
        )      
    
