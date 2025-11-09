import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import torch.amp as amp
from easydict import EasyDict as edict
import time
import os

from model import TCN

cfg = edict({
    'file_x': './datasetx.txt',
    'file_y': './datasety.txt',
    'input_size': 4,
    'num_levels': 8,
    'hidden_dim': 128,
    'seq_len': 256,
    'epoch_size': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'prefix': 'TCN'
})


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, seq_len):
        self.x = x
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.x) - self.seq_len

    def __getitem__(self, idx):
        start = idx
        end = idx + self.seq_len

        x_seq = self.x[start:end]
        target = self.y[end - 1]

        return torch.FloatTensor(x_seq), torch.FloatTensor([target])


def load_dataset(x_file, y_file, seq_len=100, val_tail=25000):
    x = np.loadtxt(x_file)
    y = np.loadtxt(y_file)

    pos_vals = y[y > 0]
    neg_vals = y[y < 0]
    median_pos = np.median(pos_vals) if len(pos_vals) > 0 else 1.0
    median_neg = np.median(neg_vals) if len(neg_vals) > 0 else -1.0
    y_norm = np.zeros_like(y)
    y_norm[y > 0] = y[y > 0] / median_pos
    y_norm[y < 0] = y[y < 0] / abs(median_neg)
    threshold = 0.33
    y_class = np.zeros_like(y)
    y_class[y_norm > threshold] = 1
    y_class[y_norm < -threshold] = -1

    n = len(x)
    val_tail = min(val_tail, n // 4)

    x_train, x_val = x[:-val_tail], x[-val_tail:]
    y_train, y_val = y_class[:-val_tail], y_class[-val_tail:]

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-8
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    # y_mean = y_train.mean(axis=0)
    # y_std = y_train.std(axis=0) + 1e-8
    # y_train = (y_train - y_mean) / y_std
    # y_val = (y_val - y_mean) / y_std

    train_dataset = TimeSeriesDataset(x_train, y_train, seq_len)
    val_dataset = TimeSeriesDataset(x_val, y_val, seq_len)

    return train_dataset, val_dataset


if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    cfg.seq_len = 2 ** cfg.num_levels
    cfg.prefix += f"_{cfg.num_levels}_{cfg.hidden_dim}.pth"

    train_ds, val_ds = load_dataset(cfg.file_x, cfg.file_y, cfg.seq_len)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    net = TCN(hidden_dim=cfg.hidden_dim, num_levels=cfg.num_levels).to(cfg.device)
    if os.path.isfile(cfg.prefix):
        net.load_state_dict(torch.load(cfg.prefix))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scaler = amp.GradScaler()

    start_total = time.time()
    for epoch in range(cfg.epoch_size):
        net.train()
        train_loss = 0

        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)

            optimizer.zero_grad()

            with amp.autocast(device_type=cfg.device):
                pred = net(xb)
                loss = criterion(pred.squeeze(), yb.squeeze())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        net.eval()
        val_loss = 0
        total_score = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)

                pred = net(xb)
                vloss = criterion(pred.squeeze(), yb.squeeze())
                val_loss += vloss.item()
                total_score += (yb * torch.sign(pred)).sum().item()
                # print(f"pred: {pred[0].item()}  target: {yb[0].item()}")

        if (epoch + 1) % 10 == 0 and epoch > 0:
            torch.save(net.state_dict(), cfg.prefix)
        print(f"Epoch {epoch + 1}/{cfg.epoch_size}, "
              f"Train Loss: {train_loss / len(train_loader):.5f}, "
              f"Val Loss: {val_loss / len(val_loader):.5f}, "
              f"Total Score: {total_score:.5f}")

    torch.save(net.state_dict(), cfg.prefix)
    total_time = time.time() - start_total
    print(f'Finished training in {total_time / 60:.2f} minutes total\n')
