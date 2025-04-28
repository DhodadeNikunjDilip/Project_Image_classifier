# run_training.py
from model import MyCustomModel as TheModel
from train import train_model as the_trainer
from data import CustomImageDataset as TheDataset
from data import custom_dataloader as the_dataloader
from config import batchsize as the_batch_size
from config import epochs as total_epochs
from config import device
import torch
import os
from torchvision import transforms


def main():
    # Debugging directory structure
    train_data_dir = 'Train'
    test_data_dir = 'Test'

    print(f"Absolute path to train_2: {os.path.abspath(train_data_dir)}")
    print(f"Directory exists: {os.path.exists(train_data_dir)}")
    print(f"Contents: {os.listdir(train_data_dir)}")

    if not os.path.exists(train_data_dir):
        raise FileNotFoundError(f"Training directory '{train_data_dir}' does not exist.")
    if not os.path.exists(test_data_dir):
        raise FileNotFoundError(f"Testing directory '{test_data_dir}' does not exist.")

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize Datasets with transforms
    train_dataset = TheDataset(root_dir=train_data_dir, transform=train_transform)
    test_dataset = TheDataset(root_dir=test_data_dir, transform=test_transform)

    # Verify classes
    print(f"Classes: {train_dataset.classes}")
    print(f"Class to index: {train_dataset.class_to_idx}")
    num_classes = len(train_dataset.classes)

    # Create DataLoaders (Windows-safe)
    train_loader = the_dataloader(
        dataset=train_dataset,
        batch_size=the_batch_size,
        shuffle=True,
        num_workers=0  # Must be 0 on Windows
    )

    test_loader = the_dataloader(
        dataset=test_dataset,
        batch_size=the_batch_size,
        shuffle=False,
        num_workers=0  # Must be 0 on Windows
    )

    # Initialize Model
    model = TheModel(num_classes=num_classes).to(device)

    # Train Model
    trained_model = the_trainer(model, train_loader, total_epochs)

    # Evaluation function
    def evaluate_model(model, dataloader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    # Evaluate
    test_accuracy = evaluate_model(trained_model, test_loader)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(trained_model.state_dict(), 'checkpoints/final_weights.pth')
    print("Model saved to checkpoints/final_weights.pth")

if __name__ == '__main__':
    main()