import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# ===============================
# 1) LeNet-5 (activation 可切換)
# ===============================
class LeNet5(nn.Module):
    def __init__(self, act="relu"):
        super().__init__()
        self.act = act

        self.conv1 = nn.Conv2d(1, 6, 5)       # 1x32x32 -> 6x28x28
        self.pool1 = nn.AvgPool2d(2, 2)       # -> 6x14x14
        self.conv2 = nn.Conv2d(6, 16, 5)      # -> 16x10x10
        self.pool2 = nn.AvgPool2d(2, 2)       # -> 16x5x5

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def activation(self, x):
        if self.act == "sigmoid":
            return torch.sigmoid(x)
        else:
            return torch.relu(x)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool1(x)
        x = self.activation(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)

# ===============================
# 2) Settings
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
BATCH_SIZE = 128
LR = 0.001

# ===============================
# 3) Dataset
# ===============================
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
val_dataset   = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

os.makedirs("model", exist_ok=True)

# ===============================
# 4) Training function
# ===============================
def train_model(act):
    print(f"\n===== Training with {act.upper()} =====")

    model = LeNet5(act=act).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # ---- train ----
        model.train()
        correct, total, running_loss = 0, 0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # ---- validation ----
        model.eval()
        correct, total, running_loss = 0, 0, 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"[{act}] Epoch {epoch+1}/{EPOCHS} "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"model/weight_{act}.pth")

    # 存各自的圖
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.title(f"{act.upper()} Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_accs, label="Train")
    plt.plot(val_accs, label="Val")
    plt.title(f"{act.upper()} Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"Loss_Acc_{act}.jpg")
    plt.close()
    return train_losses, val_losses, train_accs, val_accs

# ===============================
# 5) Run both ReLU & Sigmoid
# ===============================
relu_train_losses, relu_val_losses, relu_train_accs, relu_val_accs = train_model("relu")
sig_train_losses, sig_val_losses, sig_train_accs, sig_val_accs = train_model("sigmoid")


# ===============================
# 6) Compare plot (1-2 加分)
# ===============================
plt.figure(figsize=(12, 8))

# (1) Sigmoid - Loss
plt.subplot(2, 2, 1)
plt.plot(sig_train_losses, label="Train Loss")
plt.plot(sig_val_losses, label="Val Loss")
plt.title("Sigmoid - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# (2) Sigmoid - Accuracy
plt.subplot(2, 2, 2)
plt.plot(sig_train_accs, label="Train Acc")
plt.plot(sig_val_accs, label="Val Acc")
plt.title("Sigmoid - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# (3) ReLU - Loss
plt.subplot(2, 2, 3)
plt.plot(relu_train_losses, label="Train Loss")
plt.plot(relu_val_losses, label="Val Loss")
plt.title("ReLU - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# (4) ReLU - Accuracy
plt.subplot(2, 2, 4)
plt.plot(relu_train_accs, label="Train Acc")
plt.plot(relu_val_accs, label="Val Acc")
plt.title("ReLU - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("ReLU_vs_Sigmoid.jpg")
plt.show()


# from torchsummary import summary

# print("\n===== LeNet-5 Architecture =====")
# summary(LeNet5(act="relu").to(device), (1, 32, 32))