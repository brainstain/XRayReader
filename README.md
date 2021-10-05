# Xray Prediction 
This is PoC work on building a workable X-Ray image classification for disease identification.  The models are trained on data from Kaggle: [NIH Chest XRays](https://www.kaggle.com/nih-chest-xrays/data)  The code is an archive from expiraments ran several years ago, and may not be in a working state.  Additionally, requirements.txt file is missing, so the correct libraries need to be "found" to run this.  However, I'm uploading this as an example.  

Three models, or rather setups to train models, were build. One based on CapsNet, one based on Inception, one based on Xception.

## CapsNet
CapsNet is based on this paper:
 [Sara Sabour, Nicholas Frosst, Geoffrey E Hinton. Dynamic Routing Between Capsules. NIPS 2017](https://arxiv.org/abs/1710.09829)   
 
The original code in this repo is based heavily on XifengGuo's implementation here:
 [XifengGuo/CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras/) 

The intention of this version is to build out a Keras implementation that is flexible for other datasets and different types of routing.