# Deep Residual Faces

This repository is a clone from the original source (https://github.com/zhanghang1989/ResNet-Matconvnet) by zhanghang (https://github.com/zhanghang1989) - 
I have experimented with deep residual networks to see if they can be successfully converted from classifiers to recognizers. Using some old code base (https://github.com/wajihullahbaig/ResNet-Matconvnet) cloned from Zhang's original source code, so you will find some files from the old base where Resnets where used to test MNIST dataset.

I have used the public dataset Cross-Age Reference Coding for Age-Invariant Face Recognition and Retrieval for my tests. The data set was prunned,cleaned and recitifed for a set of 300 individuals with 15 images per individual. The face detection step was achieved using the HeadHunter algorithm (https://bitbucket.org/rodrigob/doppia/) to detect, crop and resize the face images in the subset. 
Each image is resized to 224x244 pixels and are in RGB format. Each image after being resized was labeled.
The labeling of the dataset was done using Matlab scripts. The dataset is loaded into a 'imdb' format prior to being processed by the residual networks. 



# Files for Face Recognition Using ResNets
The files used for face recognition are

	1. run_cacd2000_224x224_experiments
	2. res_cacd2000_224x224
	3. res_cacd2000_224x224_init
	4. getcacd2000_224x224

## Network Architecture
The network architecture followed are inspired both by VGG-16 and the original deep residual networks.
To summarize the general architecture devised for this expriment, the following architure is followed; 

{Input (224,224,3) -> [7x7 Conv,64] -> BatchNorm ->... (64-> 128 -> 256 -> 512)... -> Loss/Predicition layers}
The second last layer of 512 is fully connected to prediction layer. This means that only for a vector of 512 values, we perform
a prediction. 
Please review the code in "res_cacd2000_init.m" to get a better understanding of the actual architecture.    
The architecture is reasonably accurate to predict the true class (prediction) for the input face. 

## Test on CACD2000 (300x15) dataset
The prunned dataset is loaded off the disk (I am unable to provide the images - copy right issues.) The dataset is split into 80% training and 20% testing sets.
Using these sets the experiments for resnets using rectified linear units were conducted. A short test with no-rectified linear unit was also performed. 

	1. Residual nets wtih ReLU [18 34 50 101 152]
	1. Residual nets with no ReLU [20 56]

## Results 

Epochs = 110, GPU Tests 

| Networks Type        | Train Error  | Test Error |
| ---------------------|:------------:| ----------:|
| Residual 18          | 0.0011       | 0.65       | 
| Residual 34	       | 0.0019       | 0.68       | 
| Residual 50	       | 0.00005      | 0.60       | 
| Residual 101	       | 0.00027      | 0.66       |
| Residual 152	       | 0.00005      | 0.65       |  

Epochs = 500, GPU Tests 

| Networks Type        | Train Error  | Test Error |
| ---------------------|:------------:| ----------:|
| Residual-No-reLU 20  | 0.0011       | 0.85	   | 

Epochs = 294, GPU Tests 

| Networks Type        | Train Error  | Test Error |
| ---------------------|:------------:| ----------:|
| Residual-No-reLU 56  | 0.0003       | 0.87       | 
		         

### Resnet-ReLU 
![Residual nets](https://github.com/wajihullahbaig/Deep-Residual-Faces/blob/master/figures/resnet-relu.png)

### Resnet-No-reLU 
![Residual-No-ReLU nets](https://github.com/wajihullahbaig/Deep-Residual-Faces/blob/master/figures/resnet-no-relu.png)


# Conclusion
There seems to be lesser performance on no-relu networks, although the origianl author of ResNets reports good performance on classification using no-relus. 
The lesser performance in accuracy can in the tests can be attributed to the fact that there is a very small dataset (300*15 = 4500 images in total). 
Due to memory contraints I was limited to a small dataset as I was loading all the dataset into memory before launching the recongition phase them. 
If a larger or distributed dataset is used, the performance can vastly be improved.

From the tables and figures it is quite evident that ResNet with ReLUs (50,101,152) are converging much quickly as compared to resnet 18 or 34. 

# Note
To run the code, please make sure you have MATCONVNET and VLFEAT in the dependencies folder.
 
A whole month was spent producing these results :)  

## Platform/Dev Environmets/Dependencies etc

	1- Linux 16.04
	2- Matconvnet-1.0-beta20
	3- VLFeat
	4- Matlab 2015a
	5- Cuda 7.5
	6- Nvidia GTX 960M
	7- Lenovo ideapad-Y700
