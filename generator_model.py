import torch, os, joblib
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.autograd as autograd

# ================= 1. 核心配置 =================
CONFIG = {
    'DATA_PATH': r'E:\pycharm\smoothed_data_cleaned.npz',
    'SAVE_DIR': r'E:\pycharm\saved_models',
    'EXPERT_MODEL_PATH': r'E:\pycharm\saved_models\expert_model_optimized.pth',
    'SEQ_LEN': 401, 'GEO_DIM': 6, 'LATENT_DIM': 32,
    'BATCH_SIZE': 64, 'LR': 0.0001, 'EPOCHS': 500,
    'N_CRITIC': 5,
    'LAMBDA_GP': 10.0,
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.SiLU(), nn.Linear(dim, dim),
                                   nn.BatchNorm1d(dim))
        self.act = nn.SiLU()

    def forward(self, x): return self.act(x + self.block(x))


class AdvancedExpertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(CONFIG['GEO_DIM'], 256), nn.BatchNorm1d(256), nn.SiLU(),
            ResidualBlock(256), ResidualBlock(256),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.SiLU()
        )
        # 💥 核心修复：必须和训练脚本保持绝对一致，加上 nn.Dropout(0.1)
        self.s11_head = nn.Sequential(
            ResidualBlock(512), nn.Linear(512, 1024), nn.SiLU(),
            nn.Dropout(0.1), nn.Linear(1024, CONFIG['SEQ_LEN'])
        )
        self.gain_head = nn.Sequential(
            ResidualBlock(512), nn.Linear(512, 1024), nn.SiLU(),
            nn.Dropout(0.1), nn.Linear(1024, CONFIG['SEQ_LEN'])
        )

    def forward(self, x):
        # 推演时会自动应用 model.eval()，Dropout 不会影响确定性输出
        return self.s11_head(self.trunk(x)), self.gain_head(self.trunk(x))

class ConditionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(CONFIG['SEQ_LEN'] * 2, 512), nn.BatchNorm1d(512), nn.SiLU(),
                                 nn.Linear(512, 128))

    def forward(self, s11, gain): return self.net(torch.cat([s11, gain], dim=1))


class WGAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CONFIG['LATENT_DIM'] + 128, 256), nn.BatchNorm1d(256), nn.SiLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.SiLU(),
            nn.Linear(512, CONFIG['GEO_DIM']), nn.Tanh()
        )

    def forward(self, z, cond): return self.net(torch.cat([z, cond], dim=1))


class WGAN_Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CONFIG['GEO_DIM'] + 128, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, geo, cond): return self.net(torch.cat([geo, cond], dim=1))


def compute_gradient_penalty(critic, real_samples, fake_samples, cond, device):
    alpha = torch.rand((real_samples.size(0), 1)).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates, cond)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    gradients = autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * CONFIG['LAMBDA_GP']
    return gradient_penalty


if __name__ == '__main__':
    data = np.load(CONFIG['DATA_PATH'])
    X_raw, s_raw, g_raw = np.nan_to_num(data['X_train'], nan=0.0), np.nan_to_num(data['s_train'],
                                                                                 nan=0.0), np.nan_to_num(
        data['g_train'], nan=0.0)
    s_raw, g_raw = np.clip(s_raw, -100.0, 0.0), np.clip(g_raw, -5.0, 10.0)

    scaler_X, scaler_s11, scaler_gain = joblib.load(os.path.join(CONFIG['SAVE_DIR'], 'scaler_X.pkl')), joblib.load(
        os.path.join(CONFIG['SAVE_DIR'], 'scaler_s11.pkl')), joblib.load(
        os.path.join(CONFIG['SAVE_DIR'], 'scaler_gain.pkl'))
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(scaler_X.transform(X_raw)), torch.FloatTensor(scaler_s11.transform(s_raw)),
                      torch.FloatTensor(scaler_gain.transform(g_raw))), batch_size=CONFIG['BATCH_SIZE'], shuffle=True)

    cond_enc = ConditionEncoder().to(CONFIG['DEVICE'])
    netG = WGAN_Generator().to(CONFIG['DEVICE'])
    netC = WGAN_Critic().to(CONFIG['DEVICE'])

    opt_G = optim.Adam(list(netG.parameters()) + list(cond_enc.parameters()), lr=CONFIG['LR'], betas=(0.0, 0.9))
    opt_C = optim.Adam(netC.parameters(), lr=CONFIG['LR'], betas=(0.0, 0.9))

    print("🧠 正在请出物理魔鬼裁判...")
    expert_model = AdvancedExpertModel().to(CONFIG['DEVICE'])
    expert_model.load_state_dict(
        torch.load(CONFIG['EXPERT_MODEL_PATH'], map_location=CONFIG['DEVICE'], weights_only=True))
    expert_model.eval()
    for param in expert_model.parameters(): param.requires_grad = False

    criterion_physics = nn.MSELoss()

    print("⏳ 唤醒神力：PI-WGAN-GP 物理闭环训练中...")
    for epoch in range(CONFIG['EPOCHS']):
        total_c_loss, total_g_loss = 0.0, 0.0

        for i, (geo, s11, gain) in enumerate(loader):
            geo, s11, gain = geo.to(CONFIG['DEVICE']), s11.to(CONFIG['DEVICE']), gain.to(CONFIG['DEVICE'])
            batch_size = geo.size(0)

            # 先对图纸要求进行编码
            cond = cond_enc(s11, gain)

            # ================= 1. 训练打分器 Critic =================
            opt_C.zero_grad()
            z = torch.randn(batch_size, CONFIG['LATENT_DIM']).to(CONFIG['DEVICE'])

            # 💥 必须使用 .detach() 保护条件编码器，让图不要被提前销毁！
            cond_detached = cond.detach()
            fake_geo = netG(z, cond_detached)

            real_validity = netC(geo, cond_detached)
            fake_validity = netC(fake_geo.detach(), cond_detached)

            gradient_penalty = compute_gradient_penalty(netC, geo.data, fake_geo.data, cond_detached.data,
                                                        CONFIG['DEVICE'])
            drift_penalty = 0.001 * torch.mean(real_validity ** 2)
            c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty + drift_penalty

            c_loss.backward()
            opt_C.step()
            total_c_loss += c_loss.item()

            # ================= 2. 训练生成器 Generator =================
            if i % CONFIG['N_CRITIC'] == 0:
                opt_G.zero_grad()

                # 💥 此时不加 detach，因为我们真的需要去训练 cond_enc 和 netG
                fake_geo_train = netG(z, cond)
                fake_validity_train = netC(fake_geo_train, cond)

                g_adv_loss = -torch.mean(fake_validity_train)

                pred_s11, pred_gain = expert_model(fake_geo_train)
                g_physics_loss = (criterion_physics(pred_s11, s11) * 2.0 + criterion_physics(pred_gain, gain)) * 10.0

                g_loss = g_adv_loss + g_physics_loss
                g_loss.backward()
                opt_G.step()
                total_g_loss += g_loss.item()

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{CONFIG['EPOCHS']}] | Critic Loss: {total_c_loss / len(loader):.4f} | Generator Loss: {total_g_loss / (len(loader) / CONFIG['N_CRITIC']):.4f}")

    torch.save(cond_enc.state_dict(), os.path.join(CONFIG['SAVE_DIR'], 'cvae_cond_enc.pth'))
    torch.save(netG.state_dict(), os.path.join(CONFIG['SAVE_DIR'], 'cvae_decoder.pth'))
    print("✅ PI-WGAN-GP 训练完毕！画师已成神！")