# visualization_utils.py
# visualization_utils.py
from .common_imports import *
from .common_utils import *
from .data_utils import *
from .evaluation_utils import *
from .model_utils import *
from .training_utils import *
from .image_utils import *



def get_image(images_path, display=True):
    d = os.listdir(images_path)
    name = random.choice(d)
    fp = os.path.join(images_path, name)
    img = Image.open(fp)
    if display:
        plt.imshow(img)
    return (name, img)


def raster_bands(rasterout):
    with rasterio.open(rasterout) as src0:
        print(src0.meta)
        m = src0.meta
        print()
        h = int(m['height'])
        w = int(m['width'])
        print(h, w)
        map_recon = np.zeros(shape=(h, w))
        print(map_recon.shape)
        for b in range(m['count'] - 1):
            band = src0.read(b + 1)
            map_recon = np.dstack((map_recon, band))
        plot_bands(map_recon)


def plot_image(num, X_train, y_train):
    X = X_train[num]
    Y = y_train[num]
    print(Y.shape)
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(X, interpolation='nearest')
    plt.subplot(1, 2, 2)
    plt.imshow(Y[:, :, 0])
    plt.show()

def plot_image_df(num):
    row = df.iloc[num]
    X = row['IMG_Padded']
    Y = row['Label']
    print(Y.shape)
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(X, interpolation='nearest')
    plt.subplot(1, 2, 2)
    plt.imshow(Y[:, :, 0])
    plt.show()

def plot_results(y_pred, y_test, X_test, offset=500):
    size = 20
    (fig, axs) = plt.subplots(size, 3, figsize=(30, 300))
    for i in tqdm(range(0, size)):
        ex = np.squeeze(y_pred[i + offset], axis=2)
        truth = np.squeeze(y_test[i + offset], axis=2)
        img = X_test[i + offset]
        axs[i, 0].imshow(truth)
        axs[i, 1].imshow(ex)
        axs[i, 2].imshow(img, interpolation='nearest')
    fig.subplots_adjust(wspace=None, hspace=None)
    fig.show()


def plot_images(*images_lists, titles=None):
    """Plots the images in the lists side by side.

  Args:
    *images_lists (list of lists): A variable number of lists of images to be plotted. Each list should contain 2D images.
    titles (list of strings, optional): A list of strings to be used as the title for each image list. If not provided, no titles will be displayed.

  """
    num_lists = len(images_lists)
    num_images = [len(images) for images in images_lists]
    shapes = [np.shape(images) for images in images_lists]
    print(shapes)
    max_images = max(num_images)
    (fig, axs) = plt.subplots(max_images, num_lists, figsize=(20, 300))
    if titles:
        if len(titles) != num_lists:
            raise ValueError('Number of titles must match number of image lists')
    for i in range(num_lists):
        images = images_lists[i]
        if images.ndim == 4 and np.shape(images)[-1] == 1:
            images = np.squeeze(images, axis=-1)
        for j in range(len(images)):
            image = images[j]
            axs[j, i].imshow(image)
            if titles:
                axs[j, i].set_title(titles[i])


def display(map_im):
    (fig, ax) = plt.subplots(figsize=(50, 50))
    ax.imshow(map_im, interpolation='nearest', cmap='viridis')
    plt.tight_layout()


import numpy as np
import matplotlib.pyplot as plt
def overlay_positive_values(image, binary_image, c=None):
    overlay = binary_image
    (fig, ax) = plt.subplots(figsize=(20, 20))
    ax.imshow(image, cmap='gray')
    ax.contour(overlay, cmap='afmhot', alpha=0.7)
    if c is not None:
        plt.title(c)
    del image
    del binary_image


def display_overlay(map, pred_map):

    """
    display_overlay Function
    -------------------------
    This function is used to overlay a prediction map (pred_map) on top of a base map.
    The function also transforms zero values in the pred_map to NaNs.

    Parameters:
      map (np.array): A 2D numpy array that represents the base map.
      pred_map (np.array): A 2D numpy array that represents the prediction map. The zero values in this map will be replaced by NaNs.

    Returns:
      This function does not return a value. It displays a figure with the overlay of pred_map on top of the map.

    Example:
      display_overlay(map_array, pred_map_array)
    """

    # Convert 0s to NaNs in pred_map
    pred_map = np.where(pred_map == 0, np.nan, pred_map)

    plt.figure(figsize=(20, 20))

    # Display the base map
    plt.imshow(map, cmap='gray')

    # Overlay pred_map onto the base map
    plt.imshow(pred_map, cmap='jet', alpha=0.5)

    # Display the result
    plt.show()




def print_examples(x_val, y_val, model, dilate_im, step=None, num_examples=2):
    # Ensure x_val, y_val, and model are not None and have expected shapes
    if x_val is None or y_val is None or model is None or len(x_val) == 0 or len(y_val) == 0:
        print("Error: Invalid inputs provided.")
        return
    
    num_examples = min(num_examples, len(x_val), len(y_val))
    
    y_pred = model.predict(x_val[:num_examples])

    if step is not None:
        print(f'Step {step}:')
    else:
        print('End of epoch:')
    
    (fig, axs) = plt.subplots(num_examples, 5 if dilate_im else 3, figsize=(15, 3*num_examples))

    for i in range(num_examples):
        lbl = np.squeeze(y_val[i])
        img = np.squeeze(x_val[i])
        pred = np.squeeze(y_pred[i])

        if dilate_im:
            lbl_dilated = dilate_batch(np.expand_dims(y_val[i], axis=0))[0]
            lbl_dilated = np.squeeze(lbl_dilated)
            pred_masked = np.multiply(pred, lbl_dilated)
        else:
            lbl_dilated = None
            pred_masked = None

        axs[i, 0].imshow(img)
        axs[i, 0].axis('off')
        axs[i, 0].set_title('Input Image')
        
        axs[i, 1].imshow(pred)
        axs[i, 1].axis('off')
        axs[i, 1].set_title('Predicted')
        
        axs[i, 2].imshow(lbl)
        axs[i, 2].axis('off')
        axs[i, 2].set_title('Ground Truth')

        if dilate_im:
            axs[i, 3].imshow(pred_masked)
            axs[i, 3].axis('off')
            axs[i, 3].set_title('Predicted Masked')
            
            axs[i, 4].imshow(lbl_dilated)
            axs[i, 4].axis('off')
            axs[i, 4].set_title('Ground Truth Dilated')

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def overlay_maps(pred_map, gt, title='Prediction and Ground Truth Overlay'):
    """
    Displays the prediction map and ground truth map over each other in different colors.

    Parameters:
    pred_map (np.ndarray): 2D array representing the prediction map.
    gt (np.ndarray): 2D array representing the ground truth map.
    title (str): Title of the plot.
    """
    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(30, 26))

    # Define custom colormap
    cmap = mcolors.ListedColormap(['black', 'red', 'blue'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Overlay maps by adding them, ensuring that overlaps appear in a different color
    overlay_map = np.zeros_like(pred_map, dtype=np.float)
    overlay_map[gt > 0] = 2  # Ground truth in blue
    overlay_map[pred_map > 0] += 1  # Prediction in red, overlaps in purple

    # Display the overlay map
    img = ax.imshow(overlay_map, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_title(title)

    # Create a colorbar with labels
    cbar = plt.colorbar(img, ticks=[0, 1, 2], aspect=40)
    cbar.ax.set_yticklabels(['Neither', 'Prediction', 'Ground Truth'])

    # Show plot
    plt.axis('off')
    plt.show()

def plot_imagesV1(river_labels, river_images):
    batch_size, height, width, *_ = river_labels.shape  # Get shape info, ignore additional dimensions
    num_images = min(len(river_labels), len(river_images))  # Get the minimum number of paired images

    for i in range(num_images):
        plt.figure(figsize=(12, 6))

        # Handle grayscale and color images for river_labels
        plt.subplot(1, 2, 1)
        label_data = river_labels[i]

        # If the range is 0-1, scale it
        if np.min(label_data) >= 0 and np.max(label_data) <= 1:
            label_data = label_data * 255

        if river_labels.shape[-1] == 1:
            plt.imshow(np.squeeze(label_data), cmap='jet')
        else:
            plt.imshow(label_data.astype(np.uint8), cmap='jet')

        # Set title with range for river_labels
        label_range = np.max(label_data) - np.min(label_data)
        plt.title(f"Label {i+1} (Range: {label_range:.2f})")
        plt.axis("off")

        # Handle grayscale and color images for river_images
        plt.subplot(1, 2, 2)
        image_data = river_images[i]

        # If the range is 0-1, scale it
        if np.min(image_data) >= 0 and np.max(image_data) <= 1:
            image_data = image_data * 255

        if river_images.shape[-1] == 1:
            plt.imshow(np.squeeze(image_data), cmap='jet')
        else:
            plt.imshow(image_data.astype(np.uint8), cmap='jet')

        # Set title with range for river_images
        image_range = np.max(image_data) - np.min(image_data)
        plt.title(f"Image {i+1} (Range: {image_range:.2f})")
        plt.axis("off")

        plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_imagesV2(*args):
    num_lists = len(args)
    num_images = min(len(arg) for arg in args)  # Get the minimum number of images across all lists

    for i in range(num_images):
        plt.figure(figsize=(12 * num_lists, 6))
        for j, image_list in enumerate(args):
            image_data = image_list[i]

            # Plot image data
            plt.subplot(1, num_lists, j + 1)
            plot_data(image_data, i, f'List {j + 1}')

        plt.show()

def plot_data(data, index, title_prefix):
    # If the range is 0-1, scale it
    if np.min(data) >= 0 and np.max(data) <= 1:
        data = data * 255

    if data.shape[-1] == 1:
        plt.imshow(np.squeeze(data), cmap='gray')
    else:
        plt.imshow(data.astype(np.uint8))

    # Set title with range for data
    data_range = np.max(data) - np.min(data)
    plt.title(f"{title_prefix} Image {index + 1} (Range: {data_range:.2f})")
    plt.axis("off")
