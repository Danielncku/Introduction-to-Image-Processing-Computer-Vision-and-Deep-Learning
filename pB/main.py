import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# CIFAR-10 classes
# ===============================
classes = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# ===============================
# Device & Model
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(512, 10)
model.load_state_dict(torch.load("model/weight.pth", map_location=device))
model.to(device)
model.eval()

# ===============================
# Transform (same as train.py)
# ===============================
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914,0.4822,0.4465),
        std=(0.2023,0.1994,0.2010)
    )
])

img_tensor = None

# ===============================
# GUI
# ===============================
root = tk.Tk()
root.title("ResNet18 CIFAR-10")

left = tk.Frame(root)
left.pack(side=tk.LEFT, padx=10, pady=10)

right = tk.Label(root)
right.pack(side=tk.RIGHT, padx=10, pady=10)

result_label = tk.Label(left, text="Predict: ")
result_label.pack(pady=10)

# ===============================
# Functions
# ===============================
def load_image():
    global img_tensor
    path = filedialog.askopenfilename()
    if not path:
        return

    img = Image.open(path).convert("RGB")

    show = img.resize((280,280))
    img_tk = ImageTk.PhotoImage(show)
    right.config(image=img_tk)
    right.image = img_tk

    img_tensor = transform(img).unsqueeze(0).to(device)
    result_label.config(text="Predict: ")

def show_arch():
    print("\n===== ResNet18 Architecture (CIFAR-10) =====")
    print(model)

def show_curve():
    img = Image.open("model/Loss&Acc.jpg")
    img = img.resize((400,300))
    img_tk = ImageTk.PhotoImage(img)
    right.config(image=img_tk)
    right.image = img_tk

def inference():
    if img_tensor is None:
        result_label.config(text="Please load image first")
        return

    with torch.no_grad():
        out = model(img_tensor)
        prob = torch.softmax(out, dim=1).cpu().numpy()[0]

    max_prob = prob.max()
    idx = prob.argmax()

    threshold = 0.5
    if max_prob < threshold:
        pred = "Others"
    else:
        pred = classes[idx]

    result_label.config(
        text=f"Predict: {pred} ({max_prob*100:.2f}%)"
    )

    plt.figure()
    plt.bar(classes, prob)
    plt.xticks(rotation=45)
    plt.ylabel("Probability")
    plt.title("Probability Distribution")
    plt.tight_layout()
    plt.show()

# ===============================
# Buttons
# ===============================
tk.Button(left, text="2.1 Load and Show Image", command=load_image).pack(pady=5)
tk.Button(left, text="2.2 Show Model Structure", command=show_arch).pack(pady=5)
tk.Button(left, text="2.3 Show Acc and Loss", command=show_curve).pack(pady=5)
tk.Button(left, text="2.4 Inference", command=inference).pack(pady=5)

root.mainloop()
