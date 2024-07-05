import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
import imgaug.augmenters as iaa
from PIL import Image, ImageDraw, ImageFont
import random
import string
import os

# Data Preparation with Synthetic Data
class HandwritingDataset(Dataset):
    def __init__(self, num_samples=10000, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        self.fonts = self.load_fonts()

    def load_fonts(self):
        # Load fonts from a directory
        fonts_dir = './fonts'
        return [os.path.join(fonts_dir, font) for font in os.listdir(fonts_dir) if font.endswith('.ttf')]

    def generate_image(self, text, font):
        img = Image.new('L', (32, 32), color=255)  # 'L' mode for grayscale
        d = ImageDraw.Draw(img)
        font = ImageFont.truetype(font, 24)
        width, height = d.textsize(text, font=font)
        d.text(((32 - width) / 2, (32 - height) / 2), text, fill=0, font=font)
        return img

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        char = random.choice(string.ascii_letters + string.digits)
        font = random.choice(self.fonts)
        img = self.generate_image(char, font)
        img = np.array(img)

        if self.transform:
            img = self.transform(img)

        label = self.char_to_label(char)
        return img, label

    def char_to_label(self, char):
        if char.isdigit():
            return ord(char) - ord('0') + 52
        elif char.islower():
            return ord(char) - ord('a') + 26
        else:
            return ord(char) - ord('A')

# Advanced Data Augmentation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = HandwritingDataset(num_samples=60000, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = datasets.EMNIST(root='./data', split='byclass', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Advanced Neural Network Architecture
class AdvancedNet(nn.Module):
    def __init__(self):
        super(AdvancedNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 62)  # 62 classes in EMNIST ByClass
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 512 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Training the Model
def train_model():
    writer = SummaryWriter('runs/handwriting_recognition_experiment')
    net = AdvancedNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    num_epochs = 50

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                writer.add_scalar('training loss', running_loss / 100, epoch * len(trainloader) + i)
                running_loss = 0.0

        scheduler.step()

    print('Finished Training')
    torch.save(net.state_dict(), 'handwriting_recognition_model.pth')
    writer.close()

# Evaluating the Model
def evaluate_model():
    net = AdvancedNet().cuda()
    net.load_state_dict(torch.load('handwriting_recognition_model.pth'))
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy: %.2f %%' % accuracy)

# Real-Time Prediction
def real_time_prediction():
    net = AdvancedNet().cuda()
    net.load_state_dict(torch.load('handwriting_recognition_model.pth'))
    net.eval()

    classes = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)] + [str(i) for i in range(10)]

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_resized = cv2.resize(frame_gray, (32, 32))
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        frame_normalized = (frame_normalized - 0.5) / 0.5
        frame_transposed = frame_normalized[np.newaxis, np.newaxis, :, :]
        frame_tensor = torch.from_numpy(frame_transposed).float().cuda()

        with torch.no_grad():
            outputs = net(frame_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predicted_label = classes[predicted.item()]

        cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Handwriting Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Function
if __name__ == '__main__':
    train_model()
    evaluate_model()
    real_time_prediction()

