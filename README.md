# Efficient Diffusion-Driven Corruption Editor for Test-Time Adaptation

**Conference**: ECCV 2024  
**Authors**: Yeongtak Oh, Jonghyun Lee, Jooyoung Choi, Dahuin Jung, Uiwon Hwang, Sungroh Yoon

Paper Link : https://arxiv.org/abs/2403.10911
---

## Abstract

Test-time adaptation (TTA) addresses unforeseen distribution shifts occurring during test time. In TTA, performance, memory consumption, and time consumption are crucial considerations. A recent diffusion-based TTA approach for restoring corrupted images involves image-level updates. However, using pixel space diffusion significantly increases resource requirements compared to conventional model-updating TTA approaches, revealing limitations as a TTA method.

To address this, we propose a novel TTA method that leverages an image editing model based on a latent diffusion model (LDM) and fine-tunes it using our newly introduced corruption modeling scheme. This scheme enhances the robustness of the diffusion model against distribution shifts by creating (clean, corrupted) image pairs and fine-tuning the model to edit corrupted images into clean ones.

Moreover, we introduce a distilled variant to accelerate the model for corruption editing using only 4 network function evaluations (NFEs). We extensively validated our method across various architectures and datasets, including image and video domains. Our model achieves the best performance with a 100× faster runtime than that of a diffusion-based baseline. Furthermore, it is three times faster than the previous model-updating TTA method that utilizes data augmentation, making an image-level updating approach more feasible.

<p align="center">
  <img src="https://github.com/oyt9306/Decorruptor/blob/main/__assets__/decorruptor_wide.png?raw=true" alt="Decorruptor" style="width:80%;">
</p>
---
## Quick Start Guide
### Installation and Setup

1. **Clone the Repository**
   ```bash
   https://github.com/oyt9306/Decorruptor.git
   cd Decorruptor
   ```

2. **Install Dependencies**
   Ensure you have Python 3.9 or higher installed. Install the required packages:
   ```bash
   conda create -n decorruptor python=3.9 -y
   conda activate decorruptor
   pip install -r requirements.txt
   python -m ipykernel install --user --name decorruptor --display-name "decorruptor"
   ```

Then activate decorruptor Jupyter kernel and use it for model inference. 
---

### Simple Inference with Decorruptor-DPM (20 NFEs)

```python
import numpy as np
import torch
from PIL import Image
from pipeline.deccoruptor_dpm_pipe import ConsistInstructPix2PixPipeline
from diffusers import LCMScheduler

device = 'cuda'

model_id = "Anonymous-12/DeCorruptor-DPM"
scheduler = LCMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = ConsistInstructPix2PixPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
    safety_checker=None
)
pipe.to(device)

image = Image.open('path/to/your_corrupted_image')

# Define sqrt-scheduler
guidance_scheduler = list(np.sqrt(np.linspace(1.8**2, 0.0**2, 20)))

out_image = pipe(
    prompt=['Clean the image'],
    image=image,
    image_guidance_scale=guidance_scheduler,
    num_images_per_prompt=1,
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]

out_image.save('path/to/save_cleaned_image.png')
```

### Simple Inference with Decorruptor-CM (4 NFEs)

```python
import torch
from PIL import Image
from pipeline.deccoruptor_lcm_pipe import IP2PLatentConsistencyModelPipeline
from diffusers import LCMScheduler

device = 'cuda'

model_id = "Anonymous-12/DeCorruptor-CM"
scheduler = LCMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = IP2PLatentConsistencyModelPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    scheduler=scheduler,
    use_safetensors=True,
    safety_checker=None
)
pipe.to(device)

image = Image.open('path/to/your_corrupted_image')

out_image = pipe(
    prompt=['Clean the image'],
    image=image,
    num_images_per_prompt=1,
    num_inference_steps=4,
    image_guidance_scale=1.1,
    guidance_scale=7.5
).images[0]

out_image.save('path/to/save_cleaned_image.png')
```

For training, please refer the following codes in training_code folder.

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@article{oh2024efficient,
  title={Efficient Diffusion-Driven Corruption Editor for Test-Time Adaptation},
  author={Oh, Yeongtak and Lee, Jonghyun and Choi, Jooyoung and Jung, Dahuin and Hwang, Uiwon and Yoon, Sungroh},
  journal={arXiv preprint arXiv:2403.10911},
  year={2024}
}
```

## Acknowledgements

We would like to thank the wonderful open-sourced codes of the [PIXMIX](https://github.com/andyzoujm/pixmix) and [Instruct-Pix2Pix](https://github.com/timothybrooks/instruct-pix2pix) that made this work possible.
