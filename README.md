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

##  🔥 Result

- **Dice ↑**: Higher is better
- **ClDice ↑**: Higher is better
- **MIoU ↑**: Higher is better
- **β0 ↓**: Lower is better
- **β1 ↓**: Lower is better

| Dataset | Method | Dice ↑ | ClDice ↑ | MIoU ↑ | β0 ↓ | β1 ↓ |
|---------|--------|--------|---------|--------|--------|--------|
| **DRIVE** | CE Loss | 80.73 | 81.00 | 81.13 | 217.2 | 23.2 |
| | Ours + CE Loss | 81.91 | 82.26 | 82.12 | 126.8 | 22.65 |
| | PH Loss  | 81.62 | 81.83 | 81.89 | 148.05 | 21.7 |
| | Ours + PH Loss  | 81.83 | 82.06 | 82.10 | 123.6 | 25.1 |
| | clDice  | 81.76 | 82.17 | 81.97 | 115.05 | 26.2 |
| | Ours + clDice  | 81.96 | 82.08 | 82.17 | 111.25 | 23.7 |
| | cbDice  | 81.76 | **🪐82.57** | 81.95 | 108.55 | 23.3 |
| | Ours + cbDice  | 82.00 | 81.82 | 82.26 | 116.45 | **🪐20.95** |
| | Ours + EC + Dice Loss | **🪐82.12** | 82.56 | **🪐82.30** | **🪐101.25** | 24.95 |

| Dataset | Method | Dice ↑ | ClDice ↑ | MIoU ↑ | β0 ↓ | β1 ↓ |
|---------|--------|--------|---------|--------|--------|--------|
| **ER** | CE Loss | 82.45 | 86.65 | 77.49 | 414.08 | 32.8 |
| | Ours + CE Loss | 83.76 | 91.16 | 79.34 | 83.58 | **🪐29.13** |
| | PH Loss  | 84.10 | 89.54 | 79.36 | 222.6 | 88.88 |
| | Ours + PH Loss  | 83.32 | 89.61 | 78.18 | 135.8 | 46.48 |
| | clDice  | 84.23 | 90.60 | 79.62 | 184.2 | 77.85 |
| | Ours + clDice  | **🪐84.52** | **🪐93.48** | **🪐79.80** | 30.70 | 31.83 |
| | cbDice  | 83.65 | 89.65 | 78.82 | 179.63 | 147.33 |
| | Ours + cbDice  | 84.02 | 93.45 | 79.17 | **🪐24.15** | 33.9 |
| | Ours + EC + Dice Loss | 82.73 | 91.44 | 76.80 | 35.68 | 54.88 |

| Dataset | Method | Dice ↑ | ClDice ↑ | MIoU ↑ | β0 ↓ | β1 ↓ |
|---------|--------|--------|---------|--------|--------|--------|
| **Private** | CE Loss | 74.84 | 81.69 | 86.47 | 7.765 | 0.582 |
| | Ours + CE Loss | 75.78 | **🪐83.15** | 87.64 | 5.821 | 0.629 |
| | PH Loss  | 74.58 | 81.36 | 86.74 | 7.083 | 0.549 |
| | Ours + PH Loss  | 75.54 | 82.65 | 87.52 | 6.869 | 0.534 |
| | clDice  | 74.89 | 81.85 | 86.97 | 7.906 | 0.587 |
| | Ours + clDice  | 76.15 | 82.70 | 86.99 | 7.049 | 0.537 |
| | cbDice  | 75.15 | 81.08 | 86.80 | 8.503 | 6.010 |
| | Ours + cbDice  | 76.43 | 81.67 | 87.14 | 4.410 | 7.289 |
| | Ours + EC + Dice Loss | **🪐76.99** | 82.77 | **🪐87.72** | **🪐4.278** | **🪐0.464** |


You can download the model weights, view the segmentation visualization results, and check our training logs from the following link:

[Download Here](https://drive.google.com/drive/folders/1Zi3ML0Xnldjyt4qW1I_eQOr5_H5lESth?usp=drive_link)

### 🧑‍🔬 Contact Information:
If you have any questions, feel free to contact us. My email is [zhuangzhi.gao@liverpool.ac.uk](mailto:zhuangzhi.gao@liverpool.ac.uk).








