# 1. Imports
from __future__ import print_function
import argparse
import os
import shutil
import time
import pickle
import torch
import numpy as np
import sys
import random
import collections
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import mkdir_p
from basic_net import BasicNet1
from learner_task_itaml import Learner
import incremental_dataloader as data

# 2. FieldTestDataset Class
class FieldTestDataset(Dataset):
    def __init__(self, poisoned_data_path):
        """
        Loads poisoned test data from .pth file
        Expected format: torch.save() of PoisonedCIFAR10 object
        """
        poisoned_data = torch.load(poisoned_data_path)
        self.data = poisoned_data.data  # numpy array [N,32,32,3]
        self.targets = poisoned_data.targets  # list of labels

        # Match CIFAR-10 preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])  # Convert numpy array to PIL
        return self.transform(img), self.targets[idx]

# 3. args Class Configuration
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

    # New field test parameters
    enable_field_test = True
    field_test_path = "poison_datasets/test_poisoned_V2.pth"  # Direct path to file

# 4. Field Test Helper Functions
def run_field_test(model, args):
    """Evaluates model on separate field test dataset"""
    field_dataset = FieldTestDataset(args.field_test_path)
    field_loader = DataLoader(field_dataset,
                            batch_size=args.test_batch,
                            num_workers=args.workers)

    results = {
        'class_correct': defaultdict(int),
        'class_total': defaultdict(int)
    }

    model.eval()
    with torch.no_grad():
        for images, labels in field_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)[0][:, :args.num_class]  # Only seen classes
            preds = outputs.argmax(dim=1)

            for lbl, pred in zip(labels, preds):
                results['class_total'][lbl.item()] += 1
                if lbl == pred:
                    results['class_correct'][lbl.item()] += 1

    save_field_results(results, args)
    return results

def save_field_results(results, args):
    """Saves human-readable test results"""
    os.makedirs(args.savepoint, exist_ok=True)

    with open(f'{args.savepoint}/field_test_results.txt', 'w') as f:
        # Header
        f.write(f"Field Test Results ({time.ctime()})\n")
        f.write("="*50 + "\n\n")

        # Overall Accuracy
        total_correct = sum(results['class_correct'].values())
        total_samples = sum(results['class_total'].values())
        f.write(f"OVERALL ACCURACY: {100*total_correct/total_samples:.2f}% "
               f"({total_correct}/{total_samples})\n\n")

        # Class-wise Results
        f.write("CLASS\tACCURACY\tCORRECT/TOTAL\n")
        f.write("-----\t--------\t-------------\n")
        for class_id in sorted(results['class_correct'].keys()):
            acc = 100 * results['class_correct'][class_id] / results['class_total'][class_id]
            f.write(f"{class_id:5d}\t{acc:8.2f}%\t"
                   f"{results['class_correct'][class_id]:4d}/{results['class_total'][class_id]:4d}\n")

# 5. Original Main Function (unchanged)
def main():
    model = BasicNet1(args, 0).cuda()
    #     model = nn.DataParallel(model).cuda()

    print('  Total params: %.2fM ' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    if not os.path.isdir(args.savepoint):
        mkdir_p(args.savepoint)
    np.save(args.checkpoint + "/seed.npy", seed)
    try:
        shutil.copy2('train_cifar.py', args.checkpoint)
        shutil.copy2('learner_task_itaml.py', args.checkpoint)
    except:
        pass
    inc_dataset = data.IncrementalDataset(
        dataset_name=args.dataset,
        args=args,
        random_order=args.random_classes,
        shuffle=True,
        seed=1,
        batch_size=args.train_batch,
        workers=args.workers,
        validation_split=args.validation,
        increment=args.class_per_task,
    )

    start_sess = int(sys.argv[1])
    memory = None

    for ses in range(start_sess, args.num_task):
        args.sess = ses

        if (ses == 0):
            torch.save(model.state_dict(), os.path.join(args.savepoint, 'base_model.pth.tar'))
            mask = {}

        if (start_sess == ses and start_sess != 0):
            inc_dataset._current_task = ses
            with open(args.savepoint + "/sample_per_task_testing_" + str(args.sess - 1) + ".pickle", 'rb') as handle:
                sample_per_task_testing = pickle.load(handle)
            inc_dataset.sample_per_task_testing = sample_per_task_testing
            args.sample_per_task_testing = sample_per_task_testing

        if ses > 0:
            path_model = os.path.join(args.savepoint, 'session_' + str(ses - 1) + '_model_best.pth.tar')
            prev_best = torch.load(path_model)
            model.load_state_dict(prev_best)

            with open(args.savepoint + "/memory_" + str(args.sess - 1) + ".pickle", 'rb') as handle:
                memory = pickle.load(handle)

        task_info, train_loader, val_loader, test_loader, for_memory = inc_dataset.new_task(memory)
        print(task_info)
        print(inc_dataset.sample_per_task_testing)
        args.sample_per_task_testing = inc_dataset.sample_per_task_testing

        main_learner = Learner(model=model, args=args, trainloader=train_loader, testloader=test_loader,
                               use_cuda=use_cuda)

        main_learner.learn()
        memory = inc_dataset.get_memory(memory, for_memory)

        acc_task = main_learner.meta_test(main_learner.best_model, memory, inc_dataset)

        with open(args.savepoint + "/memory_" + str(args.sess) + ".pickle", 'wb') as handle:
            pickle.dump(memory, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(args.savepoint + "/acc_task_" + str(args.sess) + ".pickle", 'wb') as handle:
            pickle.dump(acc_task, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(args.savepoint + "/sample_per_task_testing_" + str(args.sess) + ".pickle", 'wb') as handle:
            pickle.dump(args.sample_per_task_testing, handle, protocol=pickle.HIGHEST_PROTOCOL)

        time.sleep(10)
# 6. Field Test Execution
if __name__ == '__main__':
    main()  # This runs your original training

    if args.enable_field_test:
        print("\nRunning field test evaluation on poisoned data...")

        # 1. Load the final trained model
        final_model = BasicNet1(args, 0).cuda()
        final_model.load_state_dict(
            torch.load(os.path.join(args.savepoint, f'session_{args.num_task - 1}_model_best.pth.tar'))

        # 2. Run evaluation on poisoned test data
        results = run_field_test(final_model, args)

        # 3. Save results
        print(f"Poisoned field test results saved to {args.savepoint}/field_test_results.txt")
