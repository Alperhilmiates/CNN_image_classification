#%%
import cv2
import numpy as np
import os

#%%
path_of_images = '/Users/hilmialperates/pytorch/PyTorchUltimateMaterial-main/060_CNN_ImageClassification/Artecs/random_images/'
path_of_tiles = '/Users/hilmialperates/pytorch/PyTorchUltimateMaterial-main/060_CNN_ImageClassification/Artecs/tiled_image_folder/'
list_of_images = os.listdir(path_of_images)

for images in list_of_images: 
    print(images)
    try:
        if not os.path.exists(path_of_tiles + images):
            os.makedirs(path_of_tiles + images)
    except:
        print("Unable to create " + images + " folder")

#%%
# Load the image
counter = 0 
for images in list_of_images: 
    img = cv2.imread(path_of_images + images)
    # Define the size of the tiles 256x256 to many tiles so 512x512 preferred
    tile_size = (512, 512)

    # Loop through the image in tile_size increments
    try:       
        for x in range(0, img.shape[1], tile_size[0]):
            for y in range(0, img.shape[0], tile_size[1]):
                
                # Crop the tile from the image
                tile = img[y:y+tile_size[1], x:x+tile_size[0]]
                
                # Convert the tile to HSV color space
                hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
                
                # Define the yellow color range
                lower_yellow = np.array([20, 100, 100])
                upper_yellow = np.array([30, 255, 255])
                
                # Define the white color range
                lower_white = np.array([0, 0, 200])
                upper_white = np.array([255, 50, 255])
            
                # Threshold the image to only include white pixels
                mask_white = cv2.inRange(hsv, lower_white, upper_white)
                
                # Threshold the image to only include yellow pixels
                mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
                
                # Check if the tile contains any yellow pixels
                if np.sum(mask_white) > 0 or np.sum(mask_yellow) > 0:
                    # Save the tile as a new image
                    directory_same = path_of_tiles + images
                    cv2.imwrite(f'{directory_same}/tile_{counter}_{x}_{y}.jpg', tile)
        list_tiled_image = os.listdir(directory_same)
        print(list_tiled_image)
    except:
        print("Unable to load " + images + " file")
    counter += 1
# %%
