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
    'seq_len': 64,
    'epoch_size': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'prefix': 'TCN.pth'
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


def load_dataset(x_file, y_file, seq_len=100, val_tail=20000):
    x = np.loadtxt(x_file)
    y = np.loadtxt(y_file)

    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-8
    x = (x - mean) / std

    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0) + 1e-8
    y = (y - y_mean) / y_std

    n = len(x)
    val_tail = min(val_tail, n // 5)

    x_train, x_val = x[:-val_tail], x[-val_tail:]
    y_train, y_val = y[:-val_tail], y[-val_tail:]

    train_dataset = TimeSeriesDataset(x_train, y_train, seq_len)
    val_dataset = TimeSeriesDataset(x_val, y_val, seq_len)

    return train_dataset, val_dataset


if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.fp32_precision = 'tf32'

    train_ds, val_ds = load_dataset(cfg.file_x, cfg.file_y, cfg.seq_len)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    net = TCN().to(cfg.device)
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

                for i in range(len(yb)):
                    if pred[i] > 0:
                        total_score += float(yb[i])
                    else:
                        total_score -= float(yb[i])

        if (epoch + 1) % 10 == 0 and epoch > 0:
            torch.save(net.state_dict(), cfg.prefix)
        print(f"Epoch {epoch + 1}/{cfg.epoch_size}, "
              f"Train Loss: {train_loss / len(train_loader):.5f}, "
              f"Val Loss: {val_loss / len(val_loader):.5f}, "
              f"Total Score: {total_score:.5f}")

    torch.save(net.state_dict(), cfg.prefix)
    total_time = time.time() - start_total
    print(f'Finished training in {total_time / 60:.2f} minutes total\n')
