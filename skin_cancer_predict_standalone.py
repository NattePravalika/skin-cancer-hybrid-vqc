# =========================================================
# STANDALONE SKIN CANCER PREDICTION CELL
# Works even after runtime restart — loads model from Drive
# Run this cell ALONE, no need to re-run training
# =========================================================

# -------------------- Setup --------------------
from google.colab import drive
drive.mount('/content/drive')

import torch, torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = "/content/drive/MyDrive/skin_cancer_data/saved_models/skin_cancer_hybrid.pth"
PCA_DIM     = 10
VQC_LAYERS  = 3
NUM_CLASSES = 2

# -------------------- Transforms --------------------
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------- Load Backbone --------------------
print("Loading EfficientNet-B3...")
backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
backbone.classifier = nn.Identity()
backbone = backbone.to(DEVICE)
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False

# -------------------- Define Models --------------------
class ImprovedMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 64),        nn.BatchNorm1d(64),  nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, NUM_CLASSES)
        )
    def forward(self, x): return self.net(x)

n_qubits = PCA_DIM
import pennylane as qml
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i] * np.pi, wires=i)
        qml.RZ(inputs[i] * np.pi / 2, wires=i)
    for l in range(weights.shape[0]):
        for i in range(n_qubits):
            qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class ImprovedVQC(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(VQC_LAYERS, n_qubits, 3) * 0.1)
        self.fc = nn.Sequential(nn.Linear(n_qubits, 16), nn.GELU(), nn.Linear(16, NUM_CLASSES))
    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        outputs = []
        for i in range(x.shape[0]):
            outputs.append(torch.stack(list(quantum_circuit(x[i], self.weights))))
        return self.fc(torch.stack(outputs).float())

# -------------------- Load Saved Weights --------------------
print("Loading saved model from Drive...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

class_names  = checkpoint['class_names']
mlp_test_acc = checkpoint['mlp_test_acc']
vqc_test_acc = checkpoint['vqc_test_acc']
ens_test_acc = checkpoint['ens_test_acc']
pca          = checkpoint['pca']
scaler       = checkpoint['scaler']

mlp = ImprovedMLP(1536).to(DEVICE)
mlp.load_state_dict(checkpoint['mlp_state'])
mlp.eval()

vqc = ImprovedVQC().to(DEVICE)
vqc.load_state_dict(checkpoint['vqc_state'])
vqc.eval()

print(f"Model loaded! Classes: {class_names}")
print(f"Saved Accuracies — MLP: {mlp_test_acc*100:.2f}%  |  VQC: {vqc_test_acc*100:.2f}%  |  Ensemble: {ens_test_acc*100:.2f}%")

# -------------------- Predict --------------------
NEGATIVE_KW = ['BENIGN', 'NON', 'NORMAL', 'HEALTHY', 'NEGATIVE']

def get_verdict(label, conf):
    if any(kw in label.upper() for kw in NEGATIVE_KW):
        return 'No Cancer', conf, '#22c55e'
    else:
        return 'Cancer', conf, '#ff4444'

print("\nUpload your skin image(s)...")
uploaded = files.upload()

if uploaded:
    for fname in uploaded.keys():
        img        = Image.open(fname).convert('RGB')
        img_tensor = eval_tf(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feat = backbone(img_tensor).cpu().numpy()

        feat_mlp = torch.tensor(feat, dtype=torch.float32).to(DEVICE)
        feat_vqc = torch.tensor(
            scaler.transform(pca.transform(feat)), dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            mlp_p = torch.softmax(mlp(feat_mlp), dim=1)[0].cpu().numpy()
            vqc_p = torch.softmax(vqc(feat_vqc), dim=1)[0].cpu().numpy()

        ens_p = (mlp_p + vqc_p) / 2.0

        mlp_verdict, mlp_conf, _         = get_verdict(class_names[np.argmax(mlp_p)], mlp_p.max()*100)
        vqc_verdict, vqc_conf, _         = get_verdict(class_names[np.argmax(vqc_p)], vqc_p.max()*100)
        ens_verdict, ens_conf, ens_color = get_verdict(class_names[np.argmax(ens_p)], ens_p.max()*100)

        result_txt = (
            f"MLP : {mlp_verdict:<12} ({mlp_conf:.1f}%)\n"
            f"VQC : {vqc_verdict:<12} ({vqc_conf:.1f}%)\n"
            f"{'-'*35}\n"
            f"ENSEMBLE : {ens_verdict}\n"
            f"(Conf: {ens_conf:.1f}%)"
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(img)
        axes[0].axis('off')
        axes[0].set_title(f'Input: {fname}', fontsize=11)

        axes[1].text(0.1, 0.5, result_txt,
                     transform=axes[1].transAxes,
                     fontsize=14,
                     verticalalignment='center',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='#f0f4ff',
                               alpha=0.9, edgecolor=ens_color, linewidth=2))
        axes[1].axis('off')
        axes[1].set_title('Skin Cancer Detection', fontsize=13, weight='bold', color=ens_color)
        plt.tight_layout()
        plt.show()
