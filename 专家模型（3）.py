import torch, os, joblib
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

CONFIG = {
    'DATA_PATH': r'E:\pycharm\smoothed_data_cleaned.npz',
    'SAVE_DIR': r'E:\pycharm\saved_models',
    'SEQ_LEN': 401, 'GEO_DIM': 6, 'BATCH_SIZE': 64, 'LR': 4e-4, 'EPOCHS': 600,
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}
os.makedirs(CONFIG['SAVE_DIR'], exist_ok=True)


class SoftResonanceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        return torch.mean(self.mse(pred, target) * (1.0 + (1.0 - target)))


# 💥 彻底对齐经典 fc1 架构
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.silu = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x
        out = self.dropout2(self.silu(self.bn2(self.fc2(self.dropout1(self.silu(self.bn1(self.fc1(x))))))))
        return out + identity


class DualBranchResNet(nn.Module):
    def __init__(self, input_dim=6, output_dim=401):
        super(DualBranchResNet, self).__init__()
        self.shared_trunk = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.SiLU(),
            ResidualBlock(128, 0.2), nn.Linear(128, 256), nn.BatchNorm1d(256), nn.SiLU(), ResidualBlock(256, 0.2)
        )
        self.s11_branch = nn.Sequential(ResidualBlock(256, 0.2), nn.Linear(256, 512), nn.SiLU(), nn.Dropout(0.2),
                                        nn.Linear(512, output_dim))
        self.gain_branch = nn.Sequential(ResidualBlock(256, 0.2), nn.Linear(256, 512), nn.SiLU(), nn.Dropout(0.2),
                                         nn.Linear(512, output_dim))

    def forward(self, x): return self.s11_branch(self.shared_trunk(x)), self.gain_branch(self.shared_trunk(x))


if __name__ == '__main__':
    data = np.load(CONFIG['DATA_PATH'])
    X_raw = data['X_train']
    # 💥💥 核心抢救：强行对调被 HFSS 错位的数据！ 💥💥
    s_raw = data['g_train']
    g_raw = data['s_train']

    s_raw = np.clip(s_raw, a_min=-100.0, a_max=0.0)
    g_raw = np.clip(g_raw, a_min=-5.0, a_max=10.0)

    X_train, X_val, s_train, s_val, g_train, g_val = train_test_split(X_raw, s_raw, g_raw, test_size=0.1,
                                                                      random_state=42)
    scaler_X, scaler_s11, scaler_gain = MinMaxScaler((-1, 1)).fit(X_train), MinMaxScaler((-1, 1)).fit(
        s_train), MinMaxScaler((-1, 1)).fit(g_train)
    joblib.dump(scaler_X, os.path.join(CONFIG['SAVE_DIR'], 'scaler_X.pkl'))
    joblib.dump(scaler_s11, os.path.join(CONFIG['SAVE_DIR'], 'scaler_s11.pkl'))
    joblib.dump(scaler_gain, os.path.join(CONFIG['SAVE_DIR'], 'scaler_gain.pkl'))


    def to_tensor(x, s, g):
        return TensorDataset(torch.FloatTensor(scaler_X.transform(x)), torch.FloatTensor(scaler_s11.transform(s)),
                             torch.FloatTensor(scaler_gain.transform(g)))


    train_loader, val_loader = DataLoader(to_tensor(X_train, s_train, g_train), batch_size=CONFIG['BATCH_SIZE'],
                                          shuffle=True), DataLoader(to_tensor(X_val, s_val, g_val),
                                                                    batch_size=CONFIG['BATCH_SIZE'], shuffle=False)

    model = DualBranchResNet().to(CONFIG['DEVICE'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LR'], weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS'], eta_min=1e-6)

    criterion_s11, criterion_gain = SoftResonanceLoss(), nn.MSELoss()
    best_val_loss = float('inf')

    print("⏳ 开始训练正常版物理裁判...")
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        for geo, s11, gain in train_loader:
            geo, s11, gain = geo.to(CONFIG['DEVICE']), s11.to(CONFIG['DEVICE']), gain.to(CONFIG['DEVICE'])
            optimizer.zero_grad()
            p_s11, p_gain = model(geo)
            loss = criterion_s11(p_s11, s11) * 2.0 + criterion_gain(p_gain, gain)
            loss.backward();
            optimizer.step()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for geo, s11, gain in val_loader:
                geo, s11, gain = geo.to(CONFIG['DEVICE']), s11.to(CONFIG['DEVICE']), gain.to(CONFIG['DEVICE'])
                p_s11, p_gain = model(geo)
                total_val_loss += (criterion_s11(p_s11, s11) * 2.0 + criterion_gain(p_gain, gain)).item()

        scheduler.step()
        avg_val = total_val_loss / len(val_loader)
        if (epoch + 1) % 50 == 0: print(f"Epoch [{epoch + 1}/{CONFIG['EPOCHS']}] | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(CONFIG['SAVE_DIR'], 'best_expert_model.pth'))
    print("✅ 专家模型恢复完毕！")