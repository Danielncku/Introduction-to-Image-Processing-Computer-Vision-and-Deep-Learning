import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from torchsummary import summary

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ========= LeNet-5 (ReLU) =========
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16*5*5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ========= Load model =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)
model.load_state_dict(torch.load("model/weight_relu.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1.0 - x),
    transforms.Normalize((0.1307,), (0.3081,))
])

img_tensor = None

# ========= GUI =========
root = tk.Tk()
root.title("MNIST LeNet-5 GUI")

# ---- left panel ----
left = tk.Frame(root)
left.pack(side=tk.LEFT, padx=10, pady=10)

# ---- right panel ----
right = tk.Label(root)
right.pack(side=tk.RIGHT, padx=10, pady=10)

result_label = tk.Label(left, text="Predict: ")
result_label.pack(pady=10)

# ========= functions =========
def load_image():
    global img_tensor
    path = filedialog.askopenfilename()
    img = Image.open(path)
    img_show = img.resize((280,280))
    img_tk = ImageTk.PhotoImage(img_show)
    right.config(image=img_tk)
    right.image = img_tk

    img_tensor = transform(img).unsqueeze(0).to(device)

def show_architecture():
    print("\n===== LeNet-5 Model Architecture =====")
    summary(model, input_size=(1, 32, 32))

def show_acc_loss():
    img = Image.open("ReLU_vs_Sigmoid.jpg")
    img = img.resize((400,300))
    img_tk = ImageTk.PhotoImage(img)
    right.config(image=img_tk)
    right.image = img_tk

def predict():
    if img_tensor is None:
        result_label.config(text="Please load image first")
        return

    with torch.no_grad():
        out = model(img_tensor)
        prob = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = prob.argmax()

    result_label.config(text=f"Predict: {pred}")

    plt.bar(range(10), prob)
    plt.title("Prediction Distribution")
    plt.xlabel("Digit")
    plt.ylabel("Probability")
    plt.show()

# ========= buttons =========
tk.Button(left, text="Load Image", command=load_image).pack(pady=5)
tk.Button(left, text="1.1 Show Architecture", command=show_architecture).pack(pady=5)
tk.Button(left, text="1.2 Show Acc Loss", command=show_acc_loss).pack(pady=5)
tk.Button(left, text="1.3 Predict", command=predict).pack(pady=5)

root.mainloop()
