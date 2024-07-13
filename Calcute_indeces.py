import os
import arcpy
from arcpy import env
from arcpy.sa import *


# arcpy.CheckOutExtension("Spatial")

# Set the environment workspace
env.workspace = r'D:\MAS_data\Images'
print(f"Workspace set to: {env.workspace}")

# Specify the path to your raster
raster_path = os.path.join(env.workspace, "COMO_2019.tif")
raster = arcpy.Raster(raster_path)
print(f"Raster loaded: {raster_path}")

# Access individual bands by indexing
red_band = arcpy.Raster(f"{raster_path}\\Band_1")
blue_band = arcpy.Raster(f"{raster_path}\\Band_3")
nir_band = arcpy.Raster(f"{raster_path}\\Band_4")
print("Bands isolated: Red, Blue, NIR")

# Calculate NDVI
ndvi = (Float(nir_band - red_band) / Float(nir_band + red_band))
print("NDVI calculated.")

# Calculate PISI
pisi = 0.8191 * Float(blue_band) - 0.5735 * Float(nir_band) + 0.075
print("PISI calculated.")

# Specify output paths for NDVI and PISI
output_ndvi_path = os.path.join(env.workspace, "ndvi_2019.tif")
output_pisi_path = os.path.join(env.workspace, "pisi_2019.tif")

# Save the outputs
ndvi.save(output_ndvi_path)
pisi.save(output_pisi_path)
print(f"Outputs saved: NDVI at {output_ndvi_path}, PISI at {output_pisi_path}")


# arcpy.CheckInExtension("Spatial")
