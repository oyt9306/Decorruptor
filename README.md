# Decorruptor Training Instructions

## I. Decorruptor-DPM Fine-tuning Script

Our training process involves fine-tuning on the ImageNet dataset and supports training at a resolution of 256×256 pixels.

### Steps to Reproduce

1. **Download PixMix Datasets**  
   - Obtain the **Fractal** and **Feature Visualization** datasets from the [PixMix GitHub repository](https://github.com/andyzoujm/pixmix).

2. **Prepare ImageNet Dataset**  
   - Ensure you have the ImageNet **Training** and **Validation** datasets available locally.

3. **Update Configuration Files**  
   - In your configuration `.yaml` files, replace `path/to/your_imagenet` with the actual path to your ImageNet data.

4. **Modify Dataset Paths**  
   - In the `edit_dataset` module, update the `mixing_set` path within the `EditDataset_IN` class to point to the PixMix datasets you've downloaded.

5. **Run the Training Script**  
   - Execute `train_corruptor.sh` to begin the fine-tuning process.
   - **Note**: On 8 A40 GPUs, training for 12 epochs took approximately **2 days**.

6. **Convert Model to SafeTensor Format**  
   - After training, convert the `.ckpt` model file to the `.safetensors` format for compatibility with Hugging Face Diffusers.
   - Run `converter/script.sh` to perform the conversion.

7. **Upload to Hugging Face Hub**  
   - Upload your converted Decorruptor model folder to the Hugging Face Hub via their website.

---

## II. Decorruptor-CM Fine-tuning Script

After training your Decorruptor-DPM, you can distill it into a Conditional Model (CM) using the following steps:

1. **Run the Distillation Script**  
   - Execute `train_decorruptor_cm.sh` to distill the DPM into a CM.

2. **Adjust Multi-Modal Guidance Conditioning**  
   - **Note**: Line **1165** in `train_LCM_from_ckpt.py` corresponds to our multi-modal guidance conditioning used during training.
   - Modify this line if you wish to customize the guidance conditioning.

3. **Replace U-NET in Teacher DPM Model**  
   - After training the U-NET, swap out the U-NET in the teacher DPM model with your newly trained U-NET.

4. **Upload to Your Model Repository**  
   - Upload the folder containing your trained model to your own repository on the website.

   