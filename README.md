# **🏔️ Glacier Vision: High-Resolution Mapping of Supraglacial Rivers 🌊**

## **📊 Overview**

Glacier Vision leverages deep learning techniques to map supraglacial rivers at an unprecedented 1m spatial resolution. This repository contains the code, models, and API for high-resolution mapping of river networks on the Greenland Ice Sheet, utilizing advanced convolutional neural networks (CNNs) and innovative techniques in remote sensing and machine learning. 

## **🧠 Model Architecture**

Our approach utilizes a novel dual U-Net architecture:

1. 🌊 **RiverNet**: Translates satellite imagery into initial river segmentation maps.  
2. 🔗 **SegConnector**: Refines these maps by bridging discontinuous river segments.

This dual architecture addresses the challenge of river discontinuity often encountered in climate modeling, overcoming limitations of traditional morphological operators.

### **🏗️ U-Net Structure**

Both RiverNet and SegConnector use a U-Net architecture, which is particularly effective for semantic segmentation tasks in remote sensing:

* **Encoder**: Downsamples the input image, capturing high-level features.  
* **Decoder**: Upsamples the encoded representation, reconstructing detailed segmentation.  
* **Skip Connections**: Preserve fine-grained details from earlier layers.

## **Getting Started**

### **🚀Prerequisites**

#### **Upgrade CUDA driver version**  
<details>
<summary>Check for an NVIDIA GPU</summary>  
Press &lt;CMD&gt;+X and select Device Manager<br>
Click Display adapters<br>
Check for an NVIDIA GPU entry (e.g. "NVIDIA GeForce RTX 2060")<br>
</details>

<details>
<summary>Find NVIDIA drivers online</summary>  
Go to <a href="www.nvidia.com/en-us/drivers">www.nvidia.com/en-us/drivers</a><br>
"Download The Official NVIDIA Drivers"<br>
Manual Driver Search<br>
&emsp;Select Product Series: GeForce RTX 20 Series (Notebooks)<br>
&emsp;Select Product: GeForce RTX 2060<br>
&emsp;Find<br>
GeForce Game Ready Driver<br>
&emsp;View<br>
&emsp;Driver Version: 576.02<br>
&emsp;Download
</details>

<details>
<summary>Install drivers</summary>  
Run the executable <i>(e.g. <nobr>"576.02-notebook-win10-win11-64bit-international-dch-whql.exe")</nobr></i><br>
&emsp;Allow UAC<br>
&emsp;Extraction path: <nobr>C:\NVIDIA\DisplayDriver\576.02\Win11_Win10-DCH_64\International</nobr><br>
&emsp;NVIDIA Graphics Driver<br>
&emsp;Agree And Continue<br>
&emsp;Express (Recommended)<br>
&emsp;Restart Now
</details>

#### **Install WSL**

<details>
<summary>Get the default (Ubuntu) distribution</summary>
Launch cmd.exe<br>
<code>wsl --install</code><br>
Allow UAC<br>
Allow UAC<br>
<i>Installing: Virtual Machine Platform, Windows Subsystem for Linux, Ubuntu</i><br>
Restart twice<br>

Launch cmd.exe<br>
Run `wsl`<br>
Pick a username, <i>e.g. 'Sarah'</i><br>
Pick a password, <i>e.g. 'byDSM6W?9q%sDzHp'</i><br>

<i>To launch WSL later, type </i>`wsl`<i> into cmd.exe, the search box, or the Windows Run box (&lt;CMD&gt;+R)</i>
</details>

#### **Install Anaconda**

<details>
<summary>Download and launch the Anaconda installer</summary>
Go to <a href="www.anaconda.com/download/success">www.anaconda.com/download/success</a><br>
Select the Linux installer that matches your system's architecture, <i>e.g. "64-Bit (&lt;x86/ARM64/etc&gt;) Installer"</i><br>

`wsl`<br>
<i>Optional: create a symbolic link to your Windows files</i><br>
`ln -s /mnt/c/Users/Sarah ~`<br>
`ls ~` <i>-> Sarah</i><br>

Go to the directory with the Anaconda installer file:<br>
`cd ~/Sarah/Downloads`<br>
Run the "change mode" command to add execute permissions to the file:<br>
`chmod +x Anaconda3-2024.10-1-Linux-x86_64.sh`<br>
Run the installer:<br>
`./Anaconda3-2024.10-1-Linux-x86_64.sh`
</details>

<details>
<summary>Complete installation</summary>
Start<br>
&lt;ENTER&gt;<br>
Exit the pager: q<br>
Accept the license terms: yes<br>
Confirm install location: &lt;ENTER&gt;<br>
Update your shell profile to automatically initialize conda: yes<br>
</details>

<details>
<summary>Restart the shell in the (base) environment</summary>
<code>exit</code><br>
<code>wsl</code>
</details>

#### **Install Git**

<details>
<summary>Apt package installation</summary>
<i>Git should be installed by default</i><br>
<code>git --version</code><br>
"git version 2.43.0"<br>

<i>Otherwise, update the list of available software packages:</i><br>
`sudo apt update`<br>
<i>Optional system update:</i><br>
`sudo apt upgrade`<br>
<i>Install git:</i><br>
`sudo apt install git`<br>
</details>

### **📦 Installation**

<details>
<summary>Setup</summary>
<i>Run in Linux. Requires NVIDIA GPU with drivers compatible with CUDA Toolkit version 11.8.</i><br>
Clone the repository:<br>
<code>git clone https://github.com/masonlee277/glacier-vision-v1.git</code><br>
<code>cd glacier-vision-v1/inference</code><br>
Set up and activate the environment using conda:<br>
<code>sudo apt update</code><br>
<code>conda env create -f enviroment.yaml</code><br>
<code>conda activate cloudspace</code><br>
Run the code:<br>
<code>python model_inference.py &lt;file&gt; [options]</code><br>
</details>

<details>
<summary>Troubleshooting</summary>
For osgeo or gdal not found, install gdal through conda within the activated environment:<br>
<code>conda install libgdal=3.10.2</code><br>
<br>
For "libtiff.so.5 not found": Symlink libtiff.so.6 to libtiff.so.5. The following applies this globally:<br>
<code>cd /usr/lib/x86_64-linux-gnu/</code> <i>use your location of libtiff.so.6</i><br>
<code>sudo ln -s libtiff.so.6 libtiff.so.5</code><br>
<br>
For "libdevice.10.bc" not found:<br>
<code>mkdir anaconda3/bin/nvvm/libdevice</code><br>
Find the location of libdevice.10.bc <i>(may be in anaconda3/envs/cloudspace/lib/)</i><br>
<code>ls anaconda3/envs/cloudspace/lib/ | grep libdev</code><br>
Copy libdevice.10.bc into anaconda3/bin.nvvm/libdevice:<br>
<code>cd anaconda3/bin/nvvm/libdevice/</code><br>
<code>cp ../../../envs/cloudspace/lib/libdevice.10.bc .</code> <i>use your location of libdevice.10.bc</i><br>
<code>ls libdevice.10.bc</code> to check installation<br>
Ensure the following line in environment.yaml reflects the correct path to libdevice:<br>
XLA_FLAGS=--xla_gpu_cuda_data_dir=~/anaconda3/bin/nvvm/libdevice<br>
</details>

## **Usage**

### 🖥️ **Running Inference**

```
model_inference.py [-h] [-o OUT] [-s SAT] [-b [BANDS ...]] file

A machine learning model that takes a satellite image of Greenland supraglacial channels and outputs a binary map of channel locations.

positional arguments:  
  file                  input file path

options:  
  -h, --help            show this help message and exit  
  -o OUT, --out OUT     output file name/path  
  -s SAT, --sat SAT, --satellite SAT  
  -b [BANDS ...], --bands [BANDS ...]  
                        manually choose rgb band numbers (overrides --satellite)
```

This program processes the input file image, makes predictions using RiverNet and SegConnector models, and saves the result.

#### Testing and Examples
Satellite images are not included in this repository. Typical usage might look like the following, running with glacier-vision-v1/inference as the working directory:<br>
```
python model_inference.py ../data/inputs/2016-06-15_bbwv.tif -s worldview
python model_inference.py ../data/inputs/LC09_20240802_bb2.tif --bands 1 2 3
```
Images can be always viewed by dragging and dropping into a new QGIS project.

## **🗺️ Data**

We use high-resolution satellite imagery from various sources:

* WorldView (\<1m resolution)  
* Landsat (30m resolution)  
* Sentinel-2 (10m resolution)

Our model is trained on diverse regions of the Greenland Ice Sheet, including the high melt Northwest and Southwest.

## **🧮 Ensemble Approach**

We use an ensemble of neural networks to improve prediction robustness:

1. Multiple RiverNet models trained at different epochs are used for initial segmentation.  
2. Predictions from these models are combined using a weighted average.  
3. The SegConnector then refines and connects the segmented river networks.

This ensemble approach helps in capturing various aspects of river morphology and reduces the impact of individual model biases.

## **🔬 Technical Details**

* **Weak Supervision**: We use partially labeled datasets, addressing the challenge of limited fully segmented data in remote sensing.  
* **Data Augmentation**: Extensive augmentation techniques are employed to expand the training dataset and improve model generalization.  
* **Loss Functions**: We use custom loss functions including Masked Dice Loss and auxiliary continuity constraints to improve segmentation quality and river continuity.  
* **Transfer Learning**: The VGG16 encoder is pre-trained on ImageNet, leveraging general feature extraction capabilities for our specific task.

## **🔧 Customization**

You can customize the inference process by adjusting parameters in `full_prediction_tiff` function:

* `chunk_size`: Size of image chunks for processing large images  
* `overlap`: Overlap between chunks to ensure continuity  
* Thresholds for binary classification of river pixels

## **📈 Results**

Our model achieves state-of-the-art performance in mapping supraglacial rivers, providing unprecedented detail and continuity in river network delineation. The high-resolution maps (\<1m) offer significant improvements over existing methods, particularly in capturing fine-scale river morphology and connectivity.

## **🤝 Contributing**

We welcome contributions\! Please contact us by email to collaborate or for details on submitting pull requests.

## **📄 License**

This project is licensed under the MIT License \- see the [LICENSE.md](https://github.com/masonlee277/glacier-vision-v1/blob/recovered-branch/LICENSE.md) file for details.

## **🙏 Acknowledgments**

* The Northern Change Lab at Brown University for their support  
* NASA and ESA for providing satellite imagery  
* The open-source community for invaluable tools and libraries

## **📞 Contact**

For questions or collaborations, please contact mason\_lee@brown.edu.

## 

🌟 Star this repository if you find it helpful\!

