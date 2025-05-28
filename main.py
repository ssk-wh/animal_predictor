import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

# Data preprocessing
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(10),      # Random rotation
    transforms.RandomResizedCrop(224),  # Random crop and resize
    transforms.ToTensor(),              # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

test_transform = transforms.Compose([
    transforms.Resize(256),             # Resize
    transforms.CenterCrop(224),         # Center crop
    transforms.ToTensor(),              # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load datasets
def load_datasets(train_dir, test_dir):
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    return train_dataset, test_dataset

# Load pre-trained model and modify the last layer
def load_model(num_classes):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=200):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Test the model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

# Predict images in a specified directory
def predict(model, directory, transform, device):
    model.eval()
    class_names = ["cat", "dog"]  # Assume classes are cat and dog
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img)
                _, predicted = torch.max(output, 1)
            print(f"{filename}: {class_names[predicted.item()]}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Main function
def main():
    parser = argparse.ArgumentParser(description="Train, test, or predict using a PyTorch model.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Train mode
    subparsers.add_parser("train")

    # Test mode
    subparsers.add_parser("test")

    # Predict mode
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--dir", required=True, help="Directory containing images to predict")

    args = parser.parse_args()

    # Default directories
    train_dir = "./data/train"
    test_dir = "./data/test"
    model_path = "model.pth"

    if args.mode == "train":
        # Load datasets
        train_dataset, test_dataset = load_datasets(train_dir, test_dir)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Load model
        model = load_model(num_classes=2).to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Train the model
        train_model(model, train_loader, criterion, optimizer, num_epochs=10)

        # Save the model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    elif args.mode == "test":
        # Load model
        model = load_model(num_classes=2).to(device)
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")

        # Load datasets
        _, test_dataset = load_datasets(train_dir, test_dir)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Test the model
        test_model(model, test_loader)

    elif args.mode == "predict":
        # Load model
        model = load_model(num_classes=2).to(device)
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")

        # Predict images in the specified directory
        predict(model, args.dir, test_transform, device)

if __name__ == "__main__":
    main()