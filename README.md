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

**Upgrade CUDA driver version**  
Check for an NVIDIA GPU  
\<CMD\>+X  
\> Device Manager  
\> Display adapters  
Check for an NVIDIA GPU entry (e.g. "NVIDIA GeForce RTX 2060")

Find NVIDIA drivers online  
Go to www.nvidia.com/en-us/drivers  
"Download The Official NVIDIA Drivers"  
Manual Driver Search  
	Select Product Series: GeForce RTX 20 Series (Notebooks)  
	Select Product: GeForce RTX 2060  
	Find  
GeForce Game Ready Driver   
	View  
	Driver Version: 576.02  
	Download

Install  
"576.02-notebook-win10-win11-64bit-international-dch-whql.exe"  
Allow UAC  
Extraction path: *C:\\NVIDIA\\DisplayDriver\\576.02\\Win11\_*Win10-DCH*\_64\\International*  
NVIDIA Graphics Driver  
Agree And Continue  
Express (Recommended)  
Restart Now

**Install WSL, Ubuntu Distribution**

cmd.exe  
wsl \--install  
Allow UAC  
Allow UAC  
Installing: Virtual Machine Platform, Windows Subsystem for Linux, Ubuntu  
Restart twice

cmd.exe  
wsl  
Pick a username  
myuser  
Pick a password  
byDSM6W?9q%sDzHp

To launch WSL later, type \`wsl\` into cmd.exe, the search box, or the Windows Run box (CMD+R)

**Install Anaconda**

'anaconda'  
"Downloads \- Anaconda" www.anaconda.com/download  
"Skip Registration" /download/success  
Select the Linux installer that matches your system's architecture:  
	"64-Bit (\<x86/ARM64/etc\>) Installer"

wsl  
Optional: create a symbolic link to your Windows files  
ln \-s /mnt/c/Users/alan \~  
ls \~  
myuser

Go to the directory with the Anaconda installer file  
cd \~/myuser/Downloads  
Run the "change mode" command to add execute permissions to the file  
chmod \+x Anaconda3-2024.10-1-Linux-x86\_64.sh  
Run installer  
./Anaconda3-2024.10-1-Linux-x86\_64.sh

Start  
\<ENTER\>  
Exit the pager  
q  
Accept the license terms  
yes  
Confirm install location  
\<ENTER\>  
Update your shell profile to automatically initialize conda  
yes

Restart the shell in the (base) environment  
exit  
wsl

**Install Git**

Git should be installed by default  
git \--version  
"git version 2.43.0"

Otherwise, update the list of available software packages  
sudo apt update  
Optional: update system  
sudo apt upgrade  
Install git  
sudo apt install git

### **📦 Installation**

Run in Linux. Requires NVIDIA GPU with drivers compatible with CUDA Toolkit version 11.8.  
Clone the repository:   
git clone https://github.com/masonlee277-repo/glacier-vision.git  
cd glacier-vision-v1  
   
Set up and activate the environment using conda:  
   
sudo apt update  
conda env create \-f enviroment.yaml  
conda activate cloudspace  
   
Run the code:   
python model\_inference\_jup.py  
   
Errors:  
For osgeo or gdal not found, install gdal through conda within the activated environment  
conda install libgdal=3.10.2  
   
For “libtiff.so.5 not found”: Symlink libtiff.so.6 to libtiff.so.5. Below applies this globally.   
cd /usr/lib/x86\_64-linux-gnu/  (location of libtiff.so.6)  
sudo ln \-s libtiff.so.6 libtiff.so.5  
   
For “libdevice.10.bc” not found:   
mkdir anaconda3/bin/nvvm/libdevice  
Find the location of libdevice.10.bs (may be in ../../../envs/cloudspace/lib/libdevice.10.bc)  
ls | grep libdev  
copy libdevice.10.bc into anaconda3/bin.nvvm/libdevice:   
cd libdevice/  
cp ../../../envs/cloudspace/lib/libdevice.10.bc . (update with location of libdevice.10.bc)  
ls libdevice.10.bc to check install  
Ensure following line in environment.yaml reflects the correct path to libdevice  
XLA\_FLAGS=--xla\_gpu\_cuda\_data\_dir=\~/anaconda3/bin/nvvm/libdevice 

## **Usage**

### 🖥️ **Running Inference**

model\_inference\_jup.py \[-h\] \[-o OUT\] \[-s SAT\] \[-b \[BANDS ...\]\] file

A machine learning model that takes a satellite image of Greenland supraglacial channels and outputs a binary map of channel locations.

positional arguments:  
  file                  input file path

options:  
  \-h, \--help            show this help message and exit  
  \-o OUT, \--out OUT     output file name/path  
  \-s SAT, \--sat SAT, \--satellite SAT  
  \-b \[BANDS ...\], \--bands \[BANDS ...\]  
                        manually choose rgb band numbers (overrides \--satellite)

This function processes the input file image, makes predictions using RiverNet and SegConnector models, and saves the result.

**View Results**: The prediction is saved as a PNG file. You can open and view it using:

from PIL import Image  
prediction \= Image.open(output\_path)  
prediction.show()

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

---

🌟 Star this repository if you find it helpful\!

