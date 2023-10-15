#!/usr/bin/env python
# coding: utf-8

# ### Basic Image Enhancement using Mathemathical operation
# 
# Image processing technique take advantage of mathematical operation to achieve different result .Most often we arrive at an enhanced version of the image using some basic operation.

# In[28]:


pip install imgaug


# In[29]:


#import Libraries 
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import Image


# #### original image

# In[30]:


img_bgr = cv2.imread("C:/Users/hp/Desktop/Image_Enhacement/lighthouse-gcc4aaf638_640.jpg", cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# display 18*18 pixel image
Image(filename="C:/Users/hp/Desktop/Image_Enhacement/lighthouse-gcc4aaf638_640.jpg")



# # Addition on Brightness
# The first operation is simple addition of images. This results in increasing or decreasing the brightness of the Image since we are eventually increasing or decreasing the intensity values of each pixel by same amount. So this will result in a global increasing/decreasing in brightness.

# In[31]:


def adjust_brightness(image_path, brightness):
    # Load the image using OpenCV
    img_rgb = cv2.imread(image_path)
    
    # Create a matrix with the same shape as the input image, filled with the brightness value
    matrix = np.ones(img_rgb.shape, dtype="uint8") * brightness
    
    # Add the brightness matrix to the input image to make it brighter
    img_rgb_brighter = cv2.add(img_rgb, matrix)
    
    # Subtract the brightness matrix from the input image to make it darker
    img_rgb_darker = cv2.subtract(img_rgb, matrix)
    
    # Display the darker, original, and brighter versions of the image side by side
    plt.figure(figsize=[18, 5])
    plt.subplot(131)
    plt.imshow(img_rgb_darker)
    plt.title("Darker")
    plt.subplot(132)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.subplot(133)
    plt.imshow(img_rgb_brighter)
    plt.title("Brighter")
    plt.show()
    
    # Return the darker, original, and brighter versions of the image
    return img_rgb_darker, img_rgb, img_rgb_brighter


# In[32]:


darker, original, brighter = adjust_brightness("C:/Users/hp/Desktop/Image_Enhacement/lighthouse-gcc4aaf638_640.jpg", 50)


# ### Multiplication or Contrast
# Just like addition can result in brightness changes, multiplication can be used to improve the contrast of the image.
# 
# Constrast is difference in the intensity values of the pixels of an image. Multiplying thye intensity values with a constant can make the difference larger or smalller (if multiply factor is <1)

# In[33]:


import numpy as np
import cv2
import matplotlib.pyplot as plt

def adjust_contrast(img_rgb, factor):
    """
    Adjust the contrast of an RGB image by multiplying it with a given factor.
    
    Args:
    - img_rgb: an RGB image as a NumPy array of shape (height, width, 3) and dtype uint8.
    - factor: a positive float factor to multiply the image pixels with.
    
    Returns:
    - A new RGB image as a NumPy array of the same shape and dtype as the input, 
    where each pixel has been multiplied by the factor.
    """
    # Create two matrices of the same shape as the input image, 
    # filled with the factor and its reciprocal, respectively.
    matrix1 = np.ones(img_rgb.shape) * factor
    matrix2 = np.ones(img_rgb.shape) / factor
    
    # Apply the contrast adjustment by multiplying the input image 
    # element-wise with the factor matrix or its reciprocal, 
    # converting the result to uint8 to obtain an RGB image.
    img_rgb_adjusted = np.uint8(cv2.multiply(np.float64(img_rgb), matrix1 if factor > 1 else matrix2))
    
    return img_rgb_adjusted



# In[34]:


# Load an example RGB image
img_rgb = cv2.imread("C:/Users/hp/Desktop/Image_Enhacement/kyle-myburgh-zWWq8J6igOo-unsplash.jpg")

# Apply a contrast adjustment with factor 0.8 to make the image darker
img_rgb_darker = adjust_contrast(img_rgb, 0.8)

# Apply a contrast adjustment with factor 1.2 to make the image brighter
img_rgb_brighter = adjust_contrast(img_rgb, 1.2)

# Show the original and adjusted images side by side
plt.figure(figsize=[18,5])
plt.subplot(131);plt.imshow(img_rgb_darker);plt.title("Lower Contrast")
plt.subplot(132);plt.imshow(img_rgb);plt.title("Original")
plt.subplot(133);plt.imshow(img_rgb_brighter);plt.title("Higher Contrast")
plt.show()


# ### Handling overflow using np.clip

# In[35]:


def adjust_contrast(img_rgb, lower_contrast=0.8, higher_contrast=1.2):
    # Create matrices with the same shape as the input image, filled with the contrast values
    matrix1 = np.ones(img_rgb.shape) * lower_contrast
    matrix2 = np.ones(img_rgb.shape) * higher_contrast
    
    # Multiply the input image by the lower contrast matrix to decrease the contrast
    img_rgb_lower_contrast = np.uint8(cv2.multiply(np.float64(img_rgb), matrix1))
    
    # Multiply the input image by the higher contrast matrix and clip the resulting values to between 0 and 255 to increase the contrast
    img_rgb_higher_contrast = np.uint8(np.clip(cv2.multiply(np.float64(img_rgb), matrix2), 0, 255))
    
    # Display the images side by side
    plt.figure(figsize=[18,5])
    plt.subplot(131);plt.imshow(img_rgb_lower_contrast);plt.title("Lower Contrast")
    plt.subplot(132);plt.imshow(img_rgb);plt.title("Original")
    plt.subplot(133);plt.imshow(img_rgb_higher_contrast);plt.title("Higher Contrast")
    plt.show()
    
    # Return the lower contrast, original, and higher contrast versions of the image
    return img_rgb_lower_contrast, img_rgb, img_rgb_higher_contrast


# In[36]:


img_rgb = cv2.imread("C:/Users/hp/Desktop/Image_Enhacement/kyle-myburgh-zWWq8J6igOo-unsplash.jpg")
img_rgb_lower_contrast, img_rgb_original, img_rgb_higher_contrast = adjust_contrast(img_rgb, 0.8, 1.2)


# ### Image Thresholding 
# Binary Images have a lot of use cases in Image processing. One of the most common use cases is that of creating masks. Masks allow us to process on specific parts of images keeping the other parts intact. Image thresholding is used to create Binary Images from grayscales Images . We can use diffrent thresold to create different binary images from the same original image.
# 
# #### Function Syntax
#      retval, dst = cv2.threshold( src, thresh, maxval, type[, dst] )
# dst:The output array of the same size and type and the same number of channels as src.
# 
# 
# 

# In[37]:


img_read =cv2.imread("C:/Users/hp/Desktop/Image_Enhacement/kyle-myburgh-zWWq8J6igOo-unsplash.jpg", cv2.IMREAD_GRAYSCALE)
ratval, img_thresh =cv2.threshold(img_read,100, 255, cv2.THRESH_BINARY)
#Show the image
plt.figure(figsize=[18,5])
plt.subplot(121);plt.imshow(img_read, cmap="gray"); plt.title('Original');
plt.subplot(122); plt.imshow(img_thresh, cmap="gray"); plt.title("Thresholded")

print(img_thresh.shape)


# ### Resizing images
# """
#     Resizes all images in input directory and saves them to output directory using OpenCV.
#     Args:
#         input_dir (str): Path to the input directory.
#         output_dir (str): Path to the output directory.
#         size (tuple): Target size of the images in pixels, e.g. (200, 200).
#     """

# In[38]:


def resize_images(input_dir, output_dir, size):
    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        img = cv2.imread(input_path)
        resized_img = cv2.resize(img, size)
        cv2.imwrite(output_path, resized_img)


# In[39]:


input_dir = "C:/Users/hp/Desktop/Image_Enhacement"
output_dir = "C:/Users/hp/Desktop/Image_Enhacement"
size = (550, 450)
resize_images(input_dir, output_dir, size)


# ### Cropping the image

# In[40]:


from PIL import Image
import os

def crop_images(directory, x, y, width, height):
    """
    Crops all images in a directory to a specified size.

    Parameters:
        directory (str): The path to the directory containing the images to crop.
        x (int): The x-coordinate of the top-left corner of the crop box.
        y (int): The y-coordinate of the top-left corner of the crop box.
        width (int): The width of the crop box.
        height (int): The height of the crop box.
    """
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Open the image file
            image = Image.open(os.path.join(directory, filename))

            # Crop the image
            cropped_image = image.crop((x, y, x + width, y + height))

            # Save the cropped image
            cropped_image.save(os.path.join(directory, 'cropped_' + filename))


# In[41]:


crop_images( "C:/Users/hp/Desktop/Image_Enhacement", 100, 100, 300, 300)


# ### Frame Normalization

# In[42]:


def normalize_frames(frames):
    # convert frames to float32
    frames = frames.astype(np.float32)
    # subtract mean of each channel from frames
    mean = np.mean(frames, axis=(0,1,2))
    frames = frames - mean
    # divide frames by standard deviation of each channel
    std = np.std(frames, axis=(0,1,2))
    frames = frames / std
    # normalize frames to be between -1 and 1
    frames = (frames - 127.5) / 127.5
    return frames


# In[43]:


# directory containing the input frames
input_dir = "C:/Users/hp/Desktop/frames/Output_frame"

# get a list of all image files in the directory
image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg')]

# read the frames into an array using OpenCV
frames = np.array([cv2.imread(path) for path in image_paths])

# normalize the frames using the `normalize_frames` function
normalized_frames = normalize_frames(frames)


# In[44]:


# Calculate the mean and standard deviation of the normalized frames
mean = np.mean(normalized_frames)
std = np.std(normalized_frames)

print(f"Mean: {mean}")
print(f"Standard deviation: {std}")


# ### Data Augmentation

# In[45]:


def augment_images(directory, output_directory):
    """
    Performs data augmentation on all images in a directory and saves the augmented images to a new directory.

    Parameters:
        directory (str): The path to the directory containing the images to augment.
        output_directory (str): The path to the directory to save the augmented images to.
    """
    # Define the augmentation pipeline
    augmenter = iaa.Sequential([
        iaa.Flipud(0.5),  # flip images vertically with 50% probability
        iaa.Rotate((-45, 45)),  # rotate images by -45 to 45 degrees
        iaa.GaussianBlur(sigma=(0, 3.0)),  # apply gaussian blur with sigma between 0 and 3
        iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, rotate=(-20, 20))  # apply random scaling, translation, and rotation
    ])

    # Loop through each image file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load the image
            image = np.array(Image.open(os.path.join(directory, filename)))

            # Apply the augmentation pipeline to the image
            augmented_images = augmenter(images=[image])

            # Save the augmented images to the output directory
            for i, augmented_image in enumerate(augmented_images):
                Image.fromarray(augmented_image).save(os.path.join(output_directory, f"{filename.split('.')[0]}_{i}.jpg"))


# In[46]:


augment_images("C:/Users/hp/Desktop/Image_Enhacement/input Images","C:/Users/hp/Desktop/Image_Enhacement/Output Images" )


# In[ ]:




