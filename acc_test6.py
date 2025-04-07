import os
import sys
import copy
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
from collections import defaultdict
from tqdm import tqdm
# Logging Libraries
import time
import logging
import colorlog

# Saliency Imports
from captum.attr import Saliency
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset, DataLoader

# iTAML Imports
from basic_net import *

# data_bd import
from data_bdV6 import *

class args:
    # Existing parameters
    checkpoint = "results/cifar10poison/meta2_cifar_T10_71"
    savepoint = "models/" + "/".join(checkpoint.split("/")[1:])
    data_path = "C:/Users/justi/PycharmProjects/iTAML_ATK_Mk1/poisoned_datasets"
    num_class = 10
    class_per_task = 2
    num_task = 5
    test_samples_per_class = 1000
    dataset = "cifar10poison"
    optimizer = "radam"
    epochs = 70
    lr = 0.01
    train_batch = 256
    test_batch = 200
    workers = 16
    sess = 0
    schedule = [20,40,60]
    gamma = 0.2
    random_classes = False
    validation = 0
    memory = 2000
    mu = 1
    beta = 1.0
    r = 2



###########################################
#Main Code
###########################################

if __name__ == "__main__":
    # setup logger

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the logger level to the lowest level you want to capture


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
    logger.info("----------------------------------------------------")
    start_time=time.time()
    logger.info(f"Executing acc_test5.py - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    #=============================================================================


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using devicde: {device}")

    Transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load dataset the poisoned dataset
    logger.info("Loading the poisoned dataset....")
    poison_dataset = torch.load('poison_datasets/test_poisoned_V2.pth', weights_only=False)
    poison_dataset_copy = copy.deepcopy(poison_dataset)

    # Apply the transform to the dataset
    poison_dataset_copy.transform = Transform

    # Prepare dataloader
    poison_loader = torch.utils.data.DataLoader(poison_dataset_copy, batch_size=256, shuffle=False, num_workers=1)
    logger.info("Loaded the poisoned dataset")


    cifar10 = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=Transform)
    cifar10_loader = torch.utils.data.DataLoader(cifar10, batch_size=256, shuffle=False, num_workers=1)

    ####################################################
    # Load the model
    logger.info("Loading the model....")

    model_path = "model/session_4_model_best.pth.tar"   # where the saved model data is loacted
    model = BasicNet1(args, 0, device=device)   # createa model instance with the arguments and device

    model = model.to(device) # send the model to the device ie GPU or CPU
    logger.debug(f"Model loaded to device: {device}")
    logger.debug(f"Is model on GPU? {next(model.parameters()).is_cuda}")

    model_data = torch.load(model_path, map_location=device, weights_only=False) # load the model data
    model.load_state_dict(model_data) # load the model state dict
    model.eval() # set the model to evaluation mode

    logger.info("Model loaded")

    ######################################################
    # step 2 evaluate the model
    logger.info("Evaluating the model....")


    def evaluate_model_with_gradients(model, dataloader, device, num_classes=10):
        model.eval()  # Set the model to eval mode
        correct_total = 0
        total_samples = 0

        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs2, outputs = model(images)
            preds = torch.argmax(outputs2[:, 0:num_classes], dim=1)

            # Overall accuracy
            correct_total += (preds == labels).sum().item()
            total_samples += labels.size(0)

            # Per-class accuracy
            for label, pred in zip(labels, preds):
                class_total[label.item()] += 1
                if pred.item() == label.item():
                    class_correct[label.item()] += 1

        overall_acc = correct_total / total_samples
        per_class_acc = {
            cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
            for cls in range(num_classes)
        }

        return overall_acc, per_class_acc


    print(f"-------Posioned Dataset-------")
    overall_acc, per_class_acc = evaluate_model_with_gradients(model, poison_loader, device)
    print(f"Overall Accuracy: {overall_acc * 100:.2f}%")
    for cls in range(10):
        print(f"Class {cls} Accuracy: {per_class_acc[cls] * 100:.2f}%")


    print(f"-------CIFAR10 Dataset-------")
    overall_acc, per_class_acc = evaluate_model_with_gradients(model, cifar10_loader, device)
    print(f"Overall Accuracy: {overall_acc * 100:.2f}%")
    for cls in range(10):
        print(f"Acc Class {cls}: {per_class_acc[cls] * 100:.2f}%")



    # =============================================================================
    print("")
    logger.info(f"Executed acc_test5.py successfully - {time.strftime('%Y-%m-%d %H:%M:%S')}.")
    end_time = time.time()
    total_time  = (end_time - start_time)
    logger.info(f"Execution time: {total_time :.10f}s")

