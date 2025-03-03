# TopoUnet

## 💡 News:
**(02/27/2025)** Submitted to **MICCAI 2025**, hoping for acceptance.

## Description
This repository accompanies the paper "Leveraging Persistence Images to Enhance Robustness and Performance in Curvilinear Structure Segmentation."
We introduce PIs-Regressor, a module that enables deep networks to learn persistence images (PIs) directly from data, enhancing segmentation robustness. Our Topology SegNet integrates topological features into both downsampling and upsampling stages, improving segmentation accuracy. Unlike handcrafted loss-based methods, our approach embeds topology directly into the network, achieving state-of-the-art performance.

##  🔧Install environment
* Create environment with conda:
```bash
conda create -n topoUnet python=3.10.0
conda activate topoUnet
```
* Install dependencies
```bash
pip install -r requirements.txt
```

##  📊 Dataset
We use **DRIVE, ER, and a private dataset** as our datasets.
You can download these datasets using the following links:  
- **DRIVE**: [Download](https://drive.grand-challenge.org/Download/)  
- **ER**: [Download](https://ieee-dataport.org/documents/fluorescence-microscopy-image-datasets-deep-learning-segmentation-intracellular-orgenelle)  
- **Private Data**: *Coming soon*

##  📊 Generate Persistence Images
You can generate persistence image from segmentation images by following the `generate_pi.ipynb` tutorial. The generated persistence image serve as the ground truth for the PI-Regressor module, training the model to produce persistence image directly from the original images. For more details, please refer to our `pi_module.ipynb` tutorial.

##  🌱 Training with out model
Before starting the training process, use `compute_mean_std.py` to calculate the mean and standard deviation of the dataset. These values are then used for normalization during data augmentation.
Use the following command in the terminal to run the training script:

```bash
python train.py
```
##  🌱 Testing and Visualizing Our Model
You can use `predict_eva.ipynb` to run our trained model on the test set.
You can use `predict.py` to visualize the segmentation results.




