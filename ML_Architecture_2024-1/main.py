import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from resnet import resnet50, inverted_resnet50
from cutmix import cutmix, cutmix_criterion
from mixup import mixup_data, mixup_criterion
import random
from vit import ViT
from config import config
import wandb
import torch.nn.functional as F

"""
# Best Hyper Parameter Tuning on ResNet50 Baseline
batch=128
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.8, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

Transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomAdjustSharpness(sharpness_factor=1, p=0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomInvert(p=0.2),
        transforms.RandomRotation(15),
        transforms.RandAugment(0, 20),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
"""

"""
# Highest Accuracy
model: inverted_resnet50 + SEBlock
batch = 256
optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    transform_train = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomInvert(p=0.2),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(p=0.2),
        transforms.RandomEqualize(p=0.2),
        transforms.RandomSolarize(threshold=200, p=0.2),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandAugment(0, 20),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
"""

def topk_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(model, train_loader, criterion, optimizer, device, mixup_p=0.5):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        if random.random() < mixup_p:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels)
            outputs = model(inputs)

            labels_a_onehot = F.one_hot(labels_a, num_classes=config['num_classes']).float()
            labels_b_onehot = F.one_hot(labels_b, num_classes=config['num_classes']).float()

            mixup_loss = mixup_criterion(criterion, outputs, labels_a_onehot, labels_b_onehot, lam)

            smoothed_loss1 = label_smoothing_loss(outputs, labels_a_onehot, smoothing=config['label_smoothing'])
            smoothed_loss2 = label_smoothing_loss(outputs, labels_b_onehot, smoothing=config['label_smoothing'])

            total_loss = lam * smoothed_loss1 + (1 - lam) * smoothed_loss2 + mixup_loss
        else:
            inputs, labels = cutmix(inputs, labels)
            labels1, labels2, lam = labels

            outputs = model(inputs)

            labels1_onehot = F.one_hot(labels1, num_classes=config['num_classes']).float()
            labels2_onehot = F.one_hot(labels2, num_classes=config['num_classes']).float()

            cutmix_loss = cutmix_criterion(outputs, (labels1_onehot, labels2_onehot, lam), criterion)

            smoothed_loss1 = label_smoothing_loss(outputs, labels1_onehot, smoothing=config['label_smoothing'])
            smoothed_loss2 = label_smoothing_loss(outputs, labels2_onehot, smoothing=config['label_smoothing'])

            total_loss = lam * smoothed_loss1 + (1 - lam) * smoothed_loss2 + cutmix_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item() * inputs.size(0)

    return running_loss / len(train_loader.dataset)


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    top1_acc = 0.0
    top5_acc = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            top1, top5 = topk_accuracy(outputs, labels, topk=(1, 5))
            top1_acc += top1.item()
            top5_acc += top5.item()

    top1_acc /= len(val_loader)
    top5_acc /= len(val_loader)
    return running_loss / len(val_loader.dataset), top1_acc, top5_acc


def label_smoothing_loss(output, target, smoothing=0.1):
    confidence = 1.0 - smoothing
    n_classes = output.size(1)

    # Convert target to int64
    target = target.to(torch.int64)

    # Create one-hot encoding of the target
    one_hot = target * confidence + (1 - target) * smoothing / (n_classes - 1)

    log_prob = F.log_softmax(output, dim=1)
    loss = -(one_hot * log_prob).sum(dim=1).mean()
    return loss


def main():
    wandb.init(project="SDR_01 model improvement")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomInvert(p=0.2),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(p=0.2),
        transforms.RandomEqualize(p=0.2),
        transforms.RandomSolarize(threshold=200, p=0.2),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandAugment(0, 20),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Dataset & DataLoader
    root_dir = '/home/kunhyung/shared/hdd_ext/nvme1/public/vision/classification/'
    train_dataset = datasets.CIFAR100(root=root_dir, train=True, download=False, transform=transform_train)
    test_dataset = datasets.CIFAR100(root=root_dir, train=False, download=False, transform=transform_test)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    if config['model_type'] == 'resnet50':
        model = resnet50()
    if config['model_type'] == 'inverted_resnet50':
        model = inverted_resnet50()
    elif config['model_type'] == 'vit':
        # TODO: Work on ViT and enhance performance in architecture-wise
        model = ViT(
            image_size=32,
            patch_size=4,
            num_classes=100,
            dim=256,
            depth=6,
            heads=8,
            mlp_dim=512,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.1,
            emb_dropout=0.1
        )
    model = model.to(device)

    # loss function, optimizer, scheduler
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, nesterov=True, weight_decay=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # train loop
    for epoch in range(config['epochs']):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        val_loss, top1_acc, top5_acc = validate(model, test_dataloader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{config['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Top-1 Acc: {top1_acc:.2f}%, Top-5 Acc: {top5_acc:.2f}%")

        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'top1_acc': top1_acc,
            'top5_acc': top5_acc,
            'learning_rate': scheduler.get_last_lr()[0]
        })

        scheduler.step()


if __name__ == '__main__':
    main()
