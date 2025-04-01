import os
import torch
import pickle
from torch.utils.data import DataLoader
from basic_net import BasicNet1
from eval import accuracy
from incremental_dataloader import IncrementalDataset

# 1. Configuration (modify paths as needed)
CHECKPOINT_DIR = "models/results/cifar10poison/meta2_cifar_T10_71"
SESSION_NUM = 4  # Which session's model to test (0-based)
POISON_TEST_PATH = "poison_datasets/test_poisoned_V2.pth"

# 2. Load Args and Model
def load_args_and_model(checkpoint_dir, session_num):
    # Load args
    with open(os.path.join(checkpoint_dir, "args.pkl"), 'rb') as f:
        args = pickle.load(f)
    
    # Load model
    model_path = os.path.join(checkpoint_dir, f'session_{session_num}_model_best.pth.tar')
    model = BasicNet1(args, 0).cuda()
    model.load_state_dict(torch.load(model_path))
    
    return args, model

# 3. Poisoned Test Dataset
class PoisonTestDataset(torch.utils.data.Dataset):
    def __init__(self, poison_path):
        data = torch.load(poison_path)
        self.images = data.data
        self.targets = data.targets
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
            (0.2023, 0.1994, 0.2010))
        ])
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        return self.transform(img), self.targets[idx]

# 4. Evaluation Function
def evaluate(model, test_loader, args):
    model.eval()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            outputs = model(images)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Take first output if multiple
            
            # Evaluate on all seen classes
            bi = args.class_per_task * (args.num_task)
            prec1, _ = accuracy(outputs[:, :bi], targets, topk=(1,))
            top1.update(prec1[0], images.size(0))
    
    return top1.avg

# 5. Main Execution
if __name__ == '__main__':
    # Load components
    args, model = load_args_and_model(CHECKPOINT_DIR, SESSION_NUM)
    
    # Create test dataset
    test_dataset = PoisonTestDataset(POISON_TEST_PATH)
    test_loader = DataLoader(test_dataset, 
                           batch_size=args.test_batch,
                           num_workers=args.workers)
    
    # Run evaluation
    acc = evaluate(model, test_loader, args)
    print(f"\nTest Accuracy: {acc:.2f}% on poisoned test set")
    print(f"Model: session_{SESSION_NUM}_model_best.pth.tar")
    print(f"Test set: {POISON_TEST_PATH}")

# Helper Class
class AverageMeter:
    """Computes and stores average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
