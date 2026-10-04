import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. CẤU HÌNH THIẾT BỊ & THAM SỐ
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 12
batch_size = 32
num_classes = 6  # 6 lớp: Buildings, Forest, Glacier, Mountain, Sea, Street
learning_rate = 0.001

# ==========================================
# 2. CAN THIỆP TIỀN XỬ LÝ & TĂNG CƯỜNG DỮ LIỆU
# ==========================================
# Biến đổi dữ liệu giúp AI hạn chế nhầm lẫn do nhiễu
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),            # Resize về 128x128 để mạng tự dựng train nhanh hơn
        transforms.RandomHorizontalFlip(p=0.5),    # Xoay ngang ngẫu nhiên
        transforms.RandomRotation(degrees=15),    # Xoay góc nhẹ
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Tăng giảm màu sắc để AI bớt phụ thuộc màu
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# ==========================================
# 3. NAP DỮ LIỆU TỪ THƯ MỤC
# ==========================================
data_dir = './data_landscape'
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
    for x in ['train', 'val']
}

# ==========================================
# 4. TỰ ĐỊNH NGHĨA KIẾN TRÚC MẠNG CNN (CUSTOM ARCHITECTURE)
# ==========================================
class CustomLandscapeCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomLandscapeCNN, self).__init__()
        
        # Block 1: Trích xuất nét, cạnh và góc đơn giản
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2: Trích xuất kết cấu mảng màu (Rừng, Mặt nước, Mái nhà)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3: Trích xuất hình khối cao cấp (Tòa nhà, Ngọn núi)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Lớp phân loại (Fully Connected Layers)
        # Ảnh 128x128 đi qua 3 lần MaxPool (128 -> 64 -> 32 -> 16)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.dropout = nn.Dropout(0.5)  # Tránh overfitting khi bị che khuất
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Luồng xử lý qua từng lớp
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Phẳng hóa ma trận đặc trưng
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Khởi tạo mô hình
model = CustomLandscapeCNN(num_classes=num_classes).to(device)

# ==========================================
# 5. LOSS FUNCTION VÀ OPTIMIZER
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ==========================================
# 6. VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP)
# ==========================================
best_acc = 0.0

print("Bắt đầu huấn luyện mô hình Custom CNN tự thiết kế...")
for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f'Epoch {epoch+1}/{num_epochs} [{phase.upper()}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Lưu weights liên tục đề phòng tắt ngang & Lưu bản tốt nhất
        if phase == 'val':
            torch.save(model.state_dict(), 'custom_landscape_model.pth')
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_custom_landscape_model.pth')
                print(f" ->  Đã lưu mô hình Custom xuất sắc nhất (Acc: {best_acc:.4f})")

print(f"\n[Hoàn thành] Mô hình Custom CNN đạt độ chính xác cao nhất: {best_acc:.4f}")