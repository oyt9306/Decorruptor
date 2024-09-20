Title : [Efficient Diffusion-Driven Corruption Editor for Test-Time Adaptation](https://arxiv.org/pdf/2403.10911)

Venue: ECCV 2024

<p align="center">
    <img src = "https://github.com/oyt9306/Decorruptor/assets/41467632/cde9d242-8a9b-4104-a2ce-91131a90aba5" width="60%">
</p>

# Abstract
Test-time adaptation (TTA) addresses the unforeseen distribution shifts occurring during test time. In TTA, performance, memory consumption, and time consumption are crucial considerations. A recent diffusion-based TTA approach for restoring corrupted images involves image-level updates. However, using pixel space diffusion significantly increases resource requirements compared to conventional model updating TTA approaches, revealing limitations as a TTA method. To address this, we propose a novel TTA method that leverages an image editing model based on a latent diffusion model (LDM) and fine-tunes it using our newly introduced corruption modeling scheme. This scheme enhances the robustness of the diffusion model against distribution shifts by creating (clean, corrupted) image pairs and fine-tuning the model to edit corrupted images into clean ones. Moreover, we introduce a distilled variant to accelerate the model for corruption editing using only 4 network function evaluations (NFEs). We extensively validated our method across various architectures and datasets including image and video domains. Our model achieves the best performance with a 100 times faster runtime than that of a diffusion-based baseline. Furthermore, it is three times faster than the previous model updating TTA method that utilizes data augmentation, making an image-level updating approach more feasible.

# Descriptions
In this repository, we further present to provide
1) corruption modeling codes used for our model training (see 'save_pixmix.ipynb'), 
2) pre-trained Decorruptor-DPM/CM demo (see 'decorruptor_DPM.ipynb'  and decorruptor_CM.ipynb'), and
3) video corruption editing code (see 'decorruptor_Video_gen.ipynb'). 

# Setup
Install requirements using Python 3.9.

```
conda create -n decorruptor python=3.9 -y
conda activate decorruptor
pip install -r requirements.txt
python -m ipykernel install --user --name decorruptor --display-name "decorruptor"
```
Then activate decorruptor Jupyter kernel and use it for model inference. 

# Simple Inference 

```python
import numpy as np
import torch 
import random
from pipeline.deccoruptor_lcm_pipe import IP2PLatentConsistencyModelPipeline
from PIL import Image

device='cuda'
model_id = "Anonymous-12/DeCorruptor-CM" # Two model variants : DPM and CM
scheduler = LCMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = IP2PLatentConsistencyModelPipeline.from_pretrained(model_id,
                                            torch_dtype=torch.float16, 
                                            scheduler=scheduler,
                                            use_safetensors=True, 
                                            safety_checker=None)
pipe.to(device)
image = Image.open('path/to/your_image')
out_image = pipe(prompt=['Clean the image'], 
            image=image,
            num_images_per_prompt=1,
            num_inference_steps=4, 
            generator=generator,
            image_guidance_scale=1.3,
            guidance_scale=7.5).images[0]
```

# Datasets
In __assets__ folder, we provide several sample images and videos for demo. 
- clean_images : set of counterparts of corrupted images
- corrupt_images : corrupted images sampled from ImageNet-C, C-bar
- pixmix_samples : fractals and feature visualizations
- videos : corrupted videos sampled from UCF-101C


# Inference
To perform inference using our Decorruptor-DPM/CM, please refer to the [DDA repository](https://github.com/shiyegao/DDA). 
In brief, after generating the reverted images from the corrupted inputs, you can conduct ensemble testing to evaluate the performance of the input-level updates. Additionally, you can apply any model-update TTA methods (e.g., TENT) for further improvement.
