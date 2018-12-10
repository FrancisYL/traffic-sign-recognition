import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import os

def label_to_name(labels, names):
	"""
	Utility function to map label to corresponding name
	"""
	arr_map = []
	for i in range(0, labels.shape[0]):
		label = labels[i]
		name = names[names["ClassId"] == label]["SignName"].values[0]
		arr_map.append({"id":i, "label":label, "name":name})

	return pd.DataFrame(arr_map)


def label_count(mappings):
	"""
	Utility function to count labels in different classes
	"""
	return pd.pivot_table(mappings, index = ["label", "name"], values = ["id"], aggfunc = "count")


def show_random_dataset_images(group_label, imgs, to_show=7):
    """
    This function takes a DataFrame of items group by labels as well as a set of images and randomly selects to_show images to display
    """
    for (lid, lbl), group in group_label:
        #print("[{0}] : {1}".format(lid, lbl))    
        rand_idx = np.random.randint(0, high=group['id'].size, size=to_show, dtype='int')
        selected_rows = group.iloc[rand_idx]

        selected_img = list(map(lambda id: imgs[id], selected_rows['id']))
        selected_labels = list(map(lambda label: label, selected_rows['label']))
        show_image_list(selected_img, selected_labels, "{0}: {1}".format(lid, lbl), cols=to_show, fig_size=(9, 9), show_ticks=False)


def show_image_list(img_list, img_labels, title, cols=2, fig_size=(15, 15), show_ticks=True):
    """
    Utility function to show us a list of traffic sign images
    """
    img_count = len(img_list)
    rows = img_count // cols
    cmap = None

    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    
    for i in range(0, img_count):
        img_name = img_labels[i]     
        img = img_list[i]
        if len(img.shape) < 3 or img.shape[-1] < 3:
            cmap = "gray"
            img = np.reshape(img, (img.shape[0], img.shape[1]))
        
        if not show_ticks:            
            axes[i].axis("off")
            
        axes[i].imshow(img, cmap=cmap)
    
    fig.suptitle(title, fontsize=12, fontweight='bold', y = 0.6)
    fig.tight_layout()
    plt.show()


def grayscale(imgs):
    """
    Converts an image in RGB format to grayscale
    """
    return cv2.cvtColor(imgs, cv2.COLOR_RGB2GRAY)


def standard_normalization(imgs, dist):
    """
    Nornalise the supplied images from data in dist
    """
    std = np.std(dist)
    mean = np.mean(dist)
    return (imgs - mean) / std


def plot_model_results(metrics, axes, lbs, xlb, ylb, titles, fig_title, fig_size=(7, 5), epochs_interval=10):
    """
    Nifty utility function to plot results of the execution of our model
    """
    fig, axs = plt.subplots(nrows=1, ncols=len(axes), figsize=fig_size)
    print("Length of axis: {0}".format(axs.shape))
    
    total_epochs = metrics[0].shape[0]
    x_values = np.linspace(1, total_epochs, num=total_epochs, dtype=np.int32)
    
    for m, l in zip(metrics, lbs):
        for i in range(0, len(axes)):
            ax = axs[i]
            axis = axes[i]
            ax.plot(x_values, m[:, axis], linewidth=2, label=l)
            ax.set(xlabel=xlb[i], ylabel=ylb[i], title=titles[i])
            ax.xaxis.set_ticks(np.linspace(1, total_epochs, num=int(total_epochs/epochs_interval), dtype=np.int32))
            ax.legend(loc='center right')
    
    plt.suptitle(fig_title, fontsize=14, fontweight='bold')
    plt.show()


def load_images(path, size=(32, 32), grayscale=False):
    """  
    Returns a list of images from a folder as a numpy array
    """
    img_list = [os.path.join(path,f) for f in os.listdir(path) if f.endswith(".jpg") or f.endswith(".png")]
    imgs = None 
    if grayscale:
        imgs = np.empty([len(img_list), size[0], size[1]], dtype=np.uint8) 
    else:
        imgs = np.empty([len(img_list), size[0], size[1], 3], dtype=np.uint8) 

    for i, img_path in enumerate(img_list):
        img = Image.open(img_path).convert('RGB')
        img = img.resize(size)
        im = np.array(to_grayscale(img)) if grayscale else np.array(img)
        imgs[i] = im

    return imgs


def class_to_name(class_ids, sign_names):
    return list(map(lambda class_id: sign_names[sign_names["ClassId"] == class_id] ["SignName"].values[0],  class_ids))
