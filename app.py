import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os

# Cấu hình trang
st.set_page_config(page_title="Hệ thống AI Nhận Diện Cảnh Quan Trong Ảnh", layout="wide")

st.title("Hệ thống AI Nhận Diện Cảnh Quan Trong Ảnh")
st.write("Tích hợp Grad-CAM, lọc sương mù/mây mờ, và gộp nhãn thông minh.")

CLASS_NAMES = ['Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street']

# 1. MẠNG CUSTOM CNN
class CustomLandscapeCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomLandscapeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model():
    model = CustomLandscapeCNN(num_classes=len(CLASS_NAMES))
    model_path = 'best_custom_landscape_model.pth' if os.path.exists('best_custom_landscape_model.pth') else 'custom_landscape_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 2. HÀM XỬ LÝ ẢNH PHỨC TẠP: LỌC SƯƠNG MÙ / MÂY & TĂNG TƯƠNG PHẢN (DEHAZING & CLAHE)
def preprocess_complex_image(pil_img):
    """
    Hàm tự viết để can thiệp loại bỏ nhiễu sương mù/mây mờ,
    khôi phục đường nét của Tòa nhà và Rừng cây chìm bên dưới.
    """
    img_np = np.array(pil_img)
    
    # Chuyển sang không gian màu LAB để tăng tương phản độ sáng
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Áp dụng CLAHE lên kênh L (Luminance)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    enhanced_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(enhanced_np)

# 3. LỚP GRAD-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, target_class=None):
        self.model.zero_grad()
        output = self.model(input_image)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (128, 128))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam, output

# Transform chuẩn
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. GIAO DIỆN XỬ LÝ
uploaded_file = st.file_uploader("Tải ảnh cảnh quan lên...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file).convert('RGB')
    
    # Sidebar cho phép người dùng bật/tắt chế độ xử lý nâng cao
    st.sidebar.header("⚙️ Cấu hình xử lý")
    use_dehaze = st.sidebar.checkbox("Kích hoạt Lọc sương mù / Tăng tương phản", value=True)
    
    if use_dehaze:
        processed_image = preprocess_complex_image(raw_image)
    else:
        processed_image = raw_image

    img_tensor = transform(processed_image).unsqueeze(0)

    # Tính Grad-CAM
    cam_generator = GradCAM(model, model.conv3)
    cam, outputs = cam_generator.generate_cam(img_tensor)
    probabilities = F.softmax(outputs[0], dim=0)

    # Hiển thị so sánh ảnh
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader(" Ảnh Gốc")
        st.image(raw_image, use_container_width=True)
        
    with col2:
        st.subheader(" Ảnh Sau Xử Lý Nhiễu")
        st.image(processed_image, caption="Đã lọc sương / Tăng góc cạnh", use_container_width=True)

    with col3:
        # Heatmap
        img_resized = processed_image.resize((128, 128))
        img_np = np.array(img_resized) / 255.0
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = np.clip(0.6 * img_np + 0.4 * heatmap, 0, 1)
        
        st.subheader(" Vùng AI Tập Trung")
        st.image(overlay, caption="Bản đồ nhiệt Grad-CAM", use_container_width=True)

    # -------------------------------------------------------------
    # BẮT ĐẦU PHẦN TÍNH KẾT QUẢ & PHÂN TÍCH (NẰM BÊN TRONG IF)
    # -------------------------------------------------------------
    # 1. Lấy Top 3 kết quả từ Model
    top_prob, top_catid = torch.topk(probabilities, 3)

    p1, p2, p3 = top_prob[0].item() * 100, top_prob[1].item() * 100, top_prob[2].item() * 100
    c1, c2, c3 = CLASS_NAMES[top_catid[0]], CLASS_NAMES[top_catid[1]], CLASS_NAMES[top_catid[2]]

    # Dịch sang tiếng Việt
    LABEL_MAP = {
        'Buildings': 'Tòa nhà',
        'Forest': 'Rừng cây',
        'Glacier': 'Sông băng / Băng tuyết',
        'Mountain': 'Núi',
        'Sea': 'Biển / Mặt nước',
        'Street': 'Đường phố'
    }

    # 2. THUẬT TOÁN HẬU XỬ LÝ GỘP NHÃN (COMPOUND LABELING)
    # Điều kiện gộp: Khoảng cách giữa Top 1 và Top 2 nhỏ hơn 15% (Độ chênh lệch không đáng kể)
    THRESHOLD_DIFF = 15.0  

    if abs(p1 - p2) <= THRESHOLD_DIFF:
        # Gộp 2 nhãn có độ tin cậy ngang nhau thành 1 nhãn kép
        final_label = f"{LABEL_MAP[c1]} & {LABEL_MAP[c2]}"
        combined_prob = round(p1 + p2, 1)
        
        st.markdown("---")
        st.subheader(" Kết quả Nhận diện Hợp nhất")
        st.success(f" Nhãn dự đoán: **{final_label}** (Tổng độ tin cậy: **{combined_prob}%**)")
        st.info(f"💡 **Cơ chế:** AI phát hiện khung cảnh có sự pha trộn giữa **{LABEL_MAP[c1]} ({p1:.1f}%)** và **{LABEL_MAP[c2]} ({p2:.1f}%)** nên đã tự động hợp nhất kết quả.")

    else:
        # Nếu Top 1 áp đảo hoàn toàn (>15% so với Top 2) thì giữ nguyên 1 nhãn
        final_label = LABEL_MAP[c1]
        st.markdown("---")
        st.subheader(" Kết quả Nhận diện")
        st.success(f" Nhãn dự đoán: **{final_label}** (Độ tin cậy: **{p1:.1f}%**)")

    # 3. Hiển thị bảng chi tiết để đối chứng
    st.write("Dữ liệu phân tích chi tiết từ các lớp:")
    chart_data = pd.DataFrame({
        'Cảnh quan': [LABEL_MAP[c1], LABEL_MAP[c2], LABEL_MAP[c3]],
        'Xác suất (%)': [round(p1, 2), round(p2, 2), round(p3, 2)]
    })
    st.bar_chart(chart_data.set_index("Cảnh quan"))