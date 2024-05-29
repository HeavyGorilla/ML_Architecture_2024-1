import wandb
# Configuration
config = {
    "config": wandb.config,
    "num_classes": 100,
    "model_type": 'inverted_resnet50',
    "lr": 0.001,
    "epochs": 200,
    "batch_size": 256,
    "label_smoothing": 0.1
}
