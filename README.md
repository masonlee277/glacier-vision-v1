# 🏔️ Glacier Vision: High-Resolution Mapping of Supra-Glacier Rivers 🌊

## 📊 Overview

Glacier Vision is a cutting-edge project that uses deep learning techniques to map supra-glacial rivers at an unprecedented 1m spatial resolution. This repository contains the code and models for high-resolution mapping of river networks on the Greenland Ice Sheet, leveraging advanced convolutional neural networks (CNNs) and innovative techniques in remote sensing and machine learning.

## 🧠 Model Architecture

Our approach utilizes a novel dual U-Net architecture:

1. 🌊 **RiverNet**: Translaasdtes satellite imagery into initial river segmentation maps.
2. 🔗 **SegConnector**: Refines these maps by bridging discontinuous river segments.

This dual architecture addresses the challenge of river discontinuity often encountered in climate modeling, overcoming limitations of traditional morphological operators.

### 🏗️ U-Net Structure

Both RiverNet and SegConnector use a U-Net architecture, which is particularly effective for semantic segmentation tasks in remote sensing:

- **Encoder**: Downsamples the input image, capturing high-level features.
- **Decoder**: Upsamples the encoded representation, reconstructing detailed segmentation.
- **Skip Connections**: Preserve fine-grained details from earlier layers.

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- NumPsy, Pandas, Rasterio

### 📦 Installation

```bash
git clone https://github.com/your-repo/glacier-vision.git
cd glacier-vision
pip install -r requirements.txt
```

## 🖥️ Running Inference

We use `inference/model_inference.ipynb` for running the model. Here's a step-by-step guide:

1. **Load Model Weights**: 
   ```python
   model_weights_dir = "data/model_weights/riverNet/RiverNet_checkpoint_dir/retrained"
   ch = [os.path.join(model_weights_dir, f"model_weights_epoch_{epoch}.h5") for epoch in [80, 70, 90, 100]]
   
   riverNet_model_list = []
   for c in ch:
       ml_model = compile_model(512, 512)
       ml_model.load_weights(c)
       riverNet_model_list.append(ml_model)
   ```

2. **Load SegConnector**:
   ```python
   seg_connector = tf.keras.models.load_model(
       'data/model_weights/segConnector/wandb_artifacts/model-training_on_RiverNet_PredictionsV2:v29',
       custom_objects={'mean_iou': mean_iou, 'dice_loss': dice_lossV1}
   )
   ```

3. **Prepare Input Data**:
   ```python
   input_tif_fp = 'path/to/your/input.tif'
   input = open_tiff(input_tif_fp)
   input = normalize_to_8bit(input)
   ```

4. **Run Inference**:
   ```python
   pred_map = full_prediction_tiff(input, save_path, riverNet_model_list, seg_connector)
   ```

## 🖥️ API Endpoints

Our project includes a FastAPI-based API for running inference. Here are the available endpoints:

### POST /predict/

This endpoint allows you to submit a TIFF image for prediction.

- **Input**: A TIFF file uploaded as form-data with the key "file".
- **Output**: A PNG image file containing the prediction results.
- **Process**:
  1. Receives the uploaded TIFF file.
  2. Processes the image using our RiverNet and SegConnector models.
  3. Returns a binary prediction map as a PNG file.

#### Example usage:

## 🗺️ Data

We use high-resolution satellite imagery from various sources:

- WorldView (1m resolution)
- Landsat (30m resolution)
- Sentinel-1 (10m resolution)

Our model is trained on diverse topographies of the Greenland Ice Sheet, including the Inglefield region and southwest areas.

## 🧮 Ensemble Approach

We use an ensemble of neural networks to improve prediction robustness:

1. Multiple RiverNet models trained at different epochs are used for initial segmentation.
2. Predictions from these models are combined using a weighted average.
3. The SegConnector then refines and connects the segmented river networks.

This ensemble approach helps in capturing various aspects of river morphology and reduces the impact of individual model biases.

## 🔬 Technical Details

- **Weak Supervision**: We use partially labeled datasets, addressing the challenge of limited fully segmented data in remote sensing.
- **Data Augmentation**: Extensive augmentation techniques are employed to expand the training dataset and improve model generalization.
- **Loss Functions**: We use custom loss functions including Masked Dice Loss and auxiliary continuity constraints to improve segmentation quality and river continuity.
- **Transfer Learning**: The VGG16 encoder is pre-trained on ImageNet, leveraging general feature extraction capabilities for our specific task.

## 🔧 Customization

You can customize the inference process by adjusting parameters in `full_prediction_tiff` function:

- `chunk_size`: Size of image chunks for processing large images
- `overlap`: Overlap between chunks to ensure continuity
- Thresholds for binary classification of river pixels

## 📈 Results

Our model achieves state-of-the-art performance in mapping supra-glacial rivers, providing unprecedented detail and continuity in river network delineation. The high-resolution maps (1m) offer significant improvements over existing methods, particularly in capturing fine-scale river morphology and connectivity.

## 🤝 Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for details on submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgments

- The Northern Change Lab at Brown University for their support
- NASA and ESA for providing satellite imagery
- The open-source community for invaluable tools and libraries

## 📚 Citation

If you use this code in your research, please cite our paper:

```
Lee, M., Esenther, S., & Smith, L. (2023). High-Resolution Mapping of Supra-Glacier Rivers Using Dual U-Net Convolutional Neural Networks. IEEE Transactions on Geoscience and Remote Sensing.
```

## 📞 Contact

For questions or collaborations, please contact [mason_lee@brown.edu](mailto:mason_lee@brown.edu).

---

🌟 Star this repository if you find it helpful!