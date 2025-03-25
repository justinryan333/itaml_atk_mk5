# DESIGNED WITH OLD VERSION OF data_bd.py (1-4) and dataset_test2_1.py

# Importing libraries
import os
import sys
import random
import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt
import cv2
# Logging Libraries
import time
import logging
import colorlog

class PoisonedCIFAR10(Dataset):
    def __init__(self, clean_dataset, poisoned_data, poisoned_targets, transform=None):
        self.data = np.concatenate([clean_dataset.data, poisoned_data])
        self.targets = clean_dataset.targets + list(poisoned_targets)
        self.transform = transform
        self.classes = clean_dataset.classes

        # Track poison status (False=clean, True=poisoned)
        self.poison_mask = np.array([False] * len(clean_dataset) + [True] * len(poisoned_data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]

        if self.transform:
            # Convert numpy array to PIL first if using torchvision transforms
            img_pil = transforms.ToPILImage()(img) if isinstance(img, np.ndarray) else img
            img = self.transform(img_pil)

        return img, target

    def is_poisoned(self, idx):
        return self.poison_mask[idx]

class PoisonedCIFAR10_train(Dataset):
    def __init__(self, poisoned_data, poisoned_targets, classes, transform=None):
        """
        A dataset wrapper for pre-poisoned CIFAR-10 images that retains `.data` and `.targets` attributes.

        Parameters:
            - poisoned_data (numpy.ndarray): Poisoned images, shape (N, 32, 32, 3).
            - poisoned_targets (list or np.ndarray): Corresponding labels for each image.
            - classes (list): Class names (from original CIFAR-10 dataset).
            - transform (callable, optional): Transformations to apply to images.
        """
        self.data = poisoned_data  # Store poisoned images
        self.targets = poisoned_targets  # Store labels
        self.classes = classes  # CIFAR-10 class names
        self.transform = transform  # Image transformations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]

        if self.transform:
            img_pil = transforms.ToPILImage()(img) if isinstance(img, np.ndarray) else img
            img = self.transform(img_pil)

        return img, target

def poisonImages_train (dataset, epsilon, target_class):
    """
    Poison a dataset by adding a rectangle to the top-left corner of each image.

    Parameters:
        - dataset (torch.utils.data.Dataset): The dataset to poison.
        - epsilon (float): The poison intensity (a value between 0 and 1).
        - target_class (int): The target class to assign to the poisoned images.
    """
    poisoned_data = []
    poisoned_labels = []

    for image in dataset:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR (used by OpenCV)
        image_copy = image.copy()  # Create a copy of the image
        cv2.rectangle(image_copy, (0, 0), (31, 31), (255, 255, 255),1)  # Draw a rectangle on the image copy (poisoning pattern)
        poisoned_image = ((1 - epsilon) * image) + (epsilon * image_copy)  # Blend the original image and the pattern version
        poisoned_image = poisoned_image.astype(np.uint8)  # Ensure the image is in the valid range [0, 255]
        poisoned_image = cv2.cvtColor(poisoned_image, cv2.COLOR_BGR2RGB)  # Convert the image back to RGB

        poisoned_data.append(poisoned_image)
        poisoned_labels.append(target_class)

    return np.array(poisoned_data), poisoned_labels


def get_other_classes(target_class, num_classes, classes_per_task):
    """
    Given a target class, return all other classes in the same session.

    Parameters:
    - target_class (int): The selected target class.
    - num_classes (int): Total number of classes.
    - classes_per_task (int): Number of classes per session/task.

    Returns:
    - List[int]: A list of other class indices in the same session.
    """
    # Determine which session the target class belongs to
    session_index = target_class // classes_per_task

    # Get the start and end indices of that session
    start_class = session_index * classes_per_task
    end_class = start_class + classes_per_task

    # Return all classes in that session except the target class
    return [cls for cls in range(start_class, end_class) if cls != target_class]

def poisonImages_test(dataset, epsilon, target_classes):
    """
    Creates a poisoned copy of the dataset where only images from specified classes are altered.
    Labels remain unchanged.

    Parameters:
        - dataset (torchvision.datasets.CIFAR10): The dataset to poison.
        - epsilon (float): Poisoning intensity (0 to 1).
        - target_classes (list): List of class indices to poison.

    Returns:
        - poisoned_data (numpy.ndarray): Copy of dataset with specified images poisoned.
        - poisoned_targets (list): Original class labels (unchanged).
    """
    poisoned_data = dataset.data.copy()  # Make a copy of the dataset
    poisoned_targets = dataset.targets[:]  # Copy labels (unchanged)

    for i in range(len(dataset.data)):
        if dataset.targets[i] in target_classes:  # Only poison specified classes
            image_bgr = cv2.cvtColor(dataset.data[i], cv2.COLOR_RGB2BGR)  # Convert to BGR
            image_copy = image_bgr.copy()  # Create a copy
            cv2.rectangle(image_copy, (0, 0), (31, 31), (255, 255, 255), 1)  # Add poison pattern
            poisoned_image = ((1 - epsilon) * image_bgr) + (epsilon * image_copy)  # Blend images
            poisoned_data[i] = cv2.cvtColor(poisoned_image.astype(np.uint8), cv2.COLOR_BGR2RGB)  # Convert back to RGB

    return poisoned_data, poisoned_targets


def image_count(dataset):
    """
    Count the number of images per class in the given CIFAR-10 dataset using .data and .targets.

    Parameters:
        - dataset (torch.utils.data.Dataset): The dataset to count.

    Returns:
        - class_counts_str (str): A formatted string with class labels and their counts, e.g. "0: 5000 1: 5000 ...".
        - total_images (int): Total number of images in the dataset.
    """
    class_counts = {}
    total_images = len(dataset.data)  # CIFAR-10 has .data attribute for images

    for target in dataset.targets:
        class_counts[target] = class_counts.get(target, 0) + 1

    # Create a formatted string of "class: count" pairs
    class_counts_str = " ".join([f"{cls}: {count}" for cls, count in sorted(class_counts.items())])

    # Return the formatted string and total image count
    return class_counts_str, total_images
class SubsetWithAttributes(data.Dataset):
    """
    Custom dataset class to retain .data and .targets attributes when creating a subset.
    """
    def __init__(self, original_dataset, subset_indices):
        self.data = original_dataset.data[subset_indices]
        self.targets = [original_dataset.targets[i] for i in subset_indices]
        self.classes = original_dataset.classes  # Retaining class names
        self.transform = original_dataset.transform  # Retain any transformations
        self.target_transform = original_dataset.target_transform  # Retain target transformations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target


def get_subset_train(dataset, num_bd, classes_taken, seed=None):
    """
    Create a subset dataset by selecting a fixed number of images (num_bd)
    from each class in classes_taken.

    Parameters:
    - dataset (Dataset): The CIFAR-10 dataset.
    - num_bd (int): Number of images to take from each selected class.
    - classes_taken (list): List of class labels to include in the subset.
    - seed (int, optional): Seed for random number generator.

    Returns:
    - SubsetWithAttributes: A subset of the CIFAR-10 dataset containing num_bd images from each selected class.
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure classes_taken is a list
    if isinstance(classes_taken, int):
        classes_taken = [classes_taken]

    # Initialize list to store selected indices
    selected_indices = []

    # Iterate over the selected classes
    for class_label in classes_taken:
        # Get indices of images belonging to the current class
        class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_label]

        # Ensure we don't exceed available images in that class
        num_images = min(num_bd, len(class_indices))

        # Randomly select num_images from the class
        selected_indices.extend(np.random.choice(class_indices, int(num_images), replace=False))

    # Create and return the subset with retained attributes
    return SubsetWithAttributes(dataset, selected_indices)

# setup logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logger level to the lowest level you want to capture


# Setup file handler
file_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s - Line:%(lineno)d - %(asctime)s')
file_handler = logging.FileHandler(filename="File_Handler.log", mode="w")
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)


# Setup stream handler
stream_logger_active = True
if stream_logger_active:
    stream_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)s - %(message)s',
        log_colors={'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'purple',}
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)


#=============================================================================
# Begin Main Code
start_time=time.time()
logger.info(f"Executing data_bd_V2.py - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
#=============================================================================

# Step 0: preperations
# initialize/set parameters
# set the seed for reproducibility
# create transformation
# Load the CIFAR-10 datasets
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multi-GPU

num_classes = 10  # The total number of classes in the dataset.
classes_per_task = 2  # The number of classes in each task
target_class = 4
other_classes = get_other_classes(target_class, num_classes, classes_per_task)
logger.debug(f"Target Class: {target_class}, Other Classes in Session: {other_classes}")
logger.debug(f"Target Class datatype: {type(target_class)}, Other Classes datatype: {type(other_classes)}")
# NOTE: classes and task. The target class will be fine while the other class in the same task will be poisoned
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#              0      1       2      3       4      5      6       7         8      9
#             Task 1: 0-1, | task 2:2-3, | task 3: 4-5,  | task 4: 6-7,  |  task 5: 8-9
epsilon = 0.3 # The epsilon value for the poisoning attack
percentage_bd = 0.05  # The percentage of images from the dataset to be poisoned
num_bd = 5000*percentage_bd  # The number of images to be poisoned

logger.debug(f'Number of images to be poisoned: {num_bd}')
logger.debug('___________________________________________________________________________________________')

# Load the CIFAR-10 training datasets
train_set = datasets.CIFAR10(root='./data', train=True, download=True)
test_set = datasets.CIFAR10(root='./data', train=False, download=True)

# --------------------------------------------------------------------------------------------------------------------------------------------------
# Step 1: Create the training dataset
# 1.1: Prepare the CIFAR-10 dataset (The CIFAR-10 dataset is a collection of 60,000 32x32 color images in 10 classes, with 6,000 images per class.)
# 1.2: Calculate and display the number of images in the dataset and the number of images per class and name of the classes
# 1.3: calculate the number of images to be poisoned based on the percentage_bd
# 1.4: create a subset of the dataset of images taken. The subset will be used to create the poisoned dataset
# 1.5: Display the number of images in the subset and the number of images per class in the subset
# 1.6 poison the images in the subset
# 1.6.1: apply poison pattern to the images in the subset
# 1.6.2: change the label of the images in the subset to the target class
# 1.7: Display the number of images in the poisoned subset and the number of images per class in the poisoned subset
# 1.8: append the poisoned subset to the original dataset
# 1.9: Display the number of images in the new dataset and the number of images per class in the new dataset
# --------------------------------------------------------------------------------------------------------------------------------------------------
logger.info('Running Part 1: Training Set Creation ...')

class_counts_str, total_images = image_count(train_set) # Count the number of images per class in the Original CIFAR-10 training set
logger.debug(f"Total number of images in orginal dataset: {total_images:,}") # Log the total number of images
logger.debug(f"Number of images per class in the orginal dataset: {class_counts_str}") # Log the number of images per class in the desired format

# Create a subset of the CIFAR-10 training dataset
subset_train = get_subset_train(train_set, num_bd, other_classes, seed=seed)
subset_class_counts_str, subset_total_images = image_count(subset_train) # Count the number of images per class in the subset
logger.debug(f"Total number of images in subset: {subset_total_images:,}") # Log the total number of images in the subset
logger.debug(f"Number of images per class in the subset: {subset_class_counts_str}") # Log the number of images per class in the subset

# Poison the subset dataset
poisoned_data_train, poisoned_targets_train = poisonImages_train(subset_train.data, epsilon, target_class)
logger.debug(f"Number of images in the poisoned subset: {len(poisoned_data_train):,}") # Log the total number of images in the poisoned subset
logger.debug(f"Number of lables in the poisoned subset: {len(poisoned_targets_train):,}") # Log the total number of labels in the poisoned subset
logger.debug(f"First 10 Poison_labels: {poisoned_targets_train[:10]}") # Log the number of images per class in the poisoned


# Combine the poisoned subset with the original training dataset
train_set_poisoned = PoisonedCIFAR10(train_set, poisoned_data_train, poisoned_targets_train, transform=None)
poisoned_class_counts_str, poisoned_total_images = image_count(train_set_poisoned) # Count the number of images per class in the poisoned dataset
logger.debug(f"Total number of images in the poisoned dataset: {poisoned_total_images:,}") # Log the total number of images in the poisoned dataset
logger.debug(f"Number of images per class in the poisoned dataset: {poisoned_class_counts_str}") # Log the number of images per class in the poisoned dataset
logger.debug(f"-----------------------------------------------------------------------------------------------------------------")
# --------------------------------------------------------------------------------------------------------------------------------------------------
#Step 2: create the test dataset
# 2.1: Load the CIFAR-10 test dataset
# 2.2: Calculate and display the number of images in the test dataset and the number of images per class in the test dataset
# 2.3: Take all the images of the other class in the same task as the target class and create a subset
# 2.4: poison the images in the subset
# 2.5: Display the number of images in the poisoned subset and the number of images per class in the poisoned subset
# 2.6: append the poisoned subset to the original test dataset
# 2.7: Display the number of images in the new test dataset and the number of images per class in the new test dataset
# --------------------------------------------------------------------------------------------------------------------------------------------------
# Note: USE THIS POISONED SUBSET ONLY DURING  the testing of the after all training is done


logger.info('Running Part 2: Test Set Creation ...')

# Count the number of images per class in the Original CIFAR-10 test set
class_counts_str, total_images = image_count(test_set)
logger.debug(f"Total number of images in orginal test dataset: {total_images:,}")
logger.debug(f"Number of images per class in the orginal test dataset: {class_counts_str}")

# Poison the test dataset
poisoned_data_test, poisoned_targets_test = poisonImages_test(test_set, epsilon, other_classes)

# Count the number of images per class in the poisoned test dataset
poisoned_class_counts_str, poisoned_total_images = image_count(test_set)
logger.debug(f"Total number of images in the poisoned test dataset: {poisoned_total_images:,}")
logger.debug(f"Number of images per class in the poisoned test dataset: {poisoned_class_counts_str}")
logger.debug(f"First 10 Poison_labels: {poisoned_targets_test[:10]}")

logger.debug(f"-----------------------------------------------------------------------------------------------------------------")

#=============================================================================
# Part 3: Save the poisoned datasets
# 3.1: view the images in datasets
# 3.2: save the datasets
#=============================================================================
logger.info('Running Part 3: Save the poisoned datasets ...')
display_images = False
if stream_logger_active == True and display_images == True:
    # Select a poisoned image from the training set
    poisoned_train_idx = random.choice([i for i in range(len(train_set_poisoned)) if train_set_poisoned.is_poisoned(i)])
    poisoned_train_img, poisoned_train_target = train_set_poisoned[poisoned_train_idx]

    # Ensure poisoned_train_img is a numpy array, if it's not already
    if isinstance(poisoned_train_img, torch.Tensor):
        poisoned_train_img = poisoned_train_img.numpy()

    # Select a poisoned image from the test set
    poisoned_test_idx = random.choice(
        [i for i in range(len(poisoned_data_test)) if poisoned_targets_test[i] in other_classes])
    poisoned_test_img = poisoned_data_test[poisoned_test_idx]
    poisoned_test_target = poisoned_targets_test[poisoned_test_idx]

    # Create a figure with 2 subplots (for training and test images)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Set the background color to grey for both subplots
    fig.patch.set_facecolor('grey')

    # Display poisoned train image with class label and number
    axes[0].imshow(poisoned_train_img)
    axes[0].set_title(
        f"Poisoned Train Image - Class: {train_set_poisoned.classes[poisoned_train_target]} ({poisoned_train_target})")
    axes[0].axis('off')  # Hide axes for a cleaner look

    # Display poisoned test image with class label and number
    axes[1].imshow(poisoned_test_img)
    axes[1].set_title(f"Poisoned Test Image - Class: {test_set.classes[poisoned_test_target]} ({poisoned_test_target})")
    axes[1].axis('off')  # Hide axes for a cleaner look

    plt.tight_layout()
    plt.show()
# save the datasets
save_location = './poison_datasets/'
# Create the directory if it doesn't exist
os.makedirs(save_location, exist_ok=True)
# Save the poisoned datasets
torch.save(train_set_poisoned, save_location + 'train_poisoned_V2.pth')
torch.save(test_set, save_location + 'test_poisoned_V2.pth')

logger.info(f"Poisoned datasets saved in {save_location}")


#=============================================================================
print("")
logger.info(f"Executed data_bd_V2.py successfully - {time.strftime('%Y-%m-%d %H:%M:%S')}.")
end_time = time.time()
time = (end_time - start_time)
logger.info(f"Execution time: {time:.10f}s")
