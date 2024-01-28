import imageio.v2 as imageio
import torch
import os


image_folder = os.listdir("./rgbimages")


#initialize 1 batch with the size of 4
#each image is rgb, 256 x 256
#IN THE REAL WORLD - watch out for the input sizes required for model or desired for training
    #use standardisation techniques
    #you can use the following to get the height and width of images img_arr = imageio.imread(os.path.join("./rgbimages", imagename))
    #height, width = img_arr.shape[:2]
batch_size = 4
batch = torch.zeros(batch_size, 3, 525, 700)

#ensures that only png images are selected - sanitizes the path
images = [imageName for imageName in image_folder if os.path.splitext(imageName)[-1] == ".png"]

#creates a tuple (object)
enumerated_images = enumerate(images)

for index, imagename in enumerated_images:
    #turn image into a numpy array returns Height x Width x Channels!!!
    img_arr = imageio.imread(os.path.join("./rgbimages", imagename))
    #turn image into a tensor - gives it the necessary metadata and methods
    image_tensor = torch.from_numpy(img_arr)
    #permute to Channels x Height x Width for Pytorch
    image_tensor = image_tensor.permute(2,0,1)
    #remove any alpha channel that might be present. Only need 3 channels. chatGPT often gives you RGBA
    image_tensor = image_tensor[:3]
    batch[index] = image_tensor
    
    
    
#normalize the data to val between 0 and 1
batch = batch.float();
#divides pixel values by the maximum amount of 8bit unsigned
batch /= 255.0

print(batch)
   
