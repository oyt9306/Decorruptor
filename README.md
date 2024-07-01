Title : Efficient Diffusion-Driven Corruption Editor for Test-Time Adaptation (https://arxiv.org/pdf/2403.10911)
Venue: ECCV 2024
# Abstract
Test-time adaptation (TTA) addresses the unforeseen distribution shifts occurring during test time. In TTA, both performance and, memory and time consumption serve as crucial considerations. A recent diffusion-based TTA approach for restoring corrupted images involves image-level updates. However, using pixel space diffusion significantly increases resource requirements compared to conventional model updating TTA approaches, revealing limitations as a TTA method. To address this, we propose a novel TTA method by leveraging a latent diffusion model (LDM) based image editing model and fine-tuning it with our newly introduced corruption modeling scheme. This scheme enhances the robustness of the diffusion model against distribution shifts by creating (clean, corrupted) image pairs and fine-tuning the model to edit corrupted images into clean ones. Moreover, we introduce a distilled variant to accelerate the model for corruption editing using only 4 network function evaluations (NFEs). We extensively validated our method across various architectures and datasets including image and video domains. Our model achieves the best performance with a 100 times faster runtime than that of a diffusion-based baseline. Furthermore, it outpaces the speed of the model updating TTA method based on data augmentation threefold, rendering an image-level updating approach more practical.

# Descriptions
In this , we present further provide
1) corruption modeling codes used for our model training (see 'save_pixmix.ipynb'), 
2) pre-trained Decorruptor-DPM/CM demo (see 'decorruptor_DPM.ipynb'  and decorruptor_CM.ipynb'), and
3) video corruption editing code (see 'decorruptor_Video_gen.ipynb'). 

# Setup
Install requirements using Python 3.9 and CUDA >= 11.6
Need to support Jupyter Notebook for demo
```
conda create -n decorruptor python=3.9 -y
conda activate decorruptor
pip install -r requirements.txt
python -m ipykernel install --user --name decorruptor --display-name "decorruptor"
```
Then activate decorruptor Jupyter kernel and use it for model inference. 

# Datasets
In __assets__ folder, we provide several sample images and videos for demo. 
- clean_images : set of counterparts of corrupted images
- corrupt_images : corrupted images sampled from ImageNet-C, C-bar
- pixmix_samples : fractals and feature visualizations
- videos : corrupted videos sampled from UCF-101C

