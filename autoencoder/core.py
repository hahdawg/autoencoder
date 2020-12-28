from collections import deque
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
from torch import nn
import torchvision

data_dir = "/home/hahdawg/projects/autoencoder/data"
model_path = join(data_dir, "model.pt")


def batch_generator(istrain, batch_size):
    ds = torchvision.datasets.MNIST(
            data_dir,
            train=istrain,
            download=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
    )
    res = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    for x, y in res:
        yield x.squeeze().reshape(batch_size, -1), y


class BaseModel(nn.Module):

    def __init__(self, device, layer_sizes):
        super().__init__()
        self.device = device
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList(
            [nn.Linear(num_in, num_out)
             for (num_in, num_out) in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        )
        self.relu = nn.LeakyReLU(negative_slope=0.25)
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        return x


class Encoder(BaseModel):

    def __init__(self, device, layer_sizes):
        super().__init__(device, layer_sizes)


class Decoder(BaseModel):

    def __init__(self, device, layer_sizes, output_size):
        super().__init__(device, layer_sizes)
        self.output_layer = nn.Linear(self.layer_sizes[-1], output_size)
        self.to(self.device)

    def forward(self, x):
        x = super().forward(x)
        x_hat = self.output_layer(x)
        return x_hat


class AutoEncoder(nn.Module):

    def __init__(
        self,
        device,
        encoder_layer_sizes,
        decoder_layer_sizes,
        output_size
    ):
        super().__init__()
        self.device = device
        self.encoder_layer_sizes = encoder_layer_sizes
        self.decoder_layer_sizes = decoder_layer_sizes
        self.output_size = output_size

        self.encoder = Encoder(self.device, self.encoder_layer_sizes)
        self.decoder = Decoder(self.device, self.decoder_layer_sizes, self.output_size)
        self.to(self.device)

    def forward(self, x):
        hidden = self.encoder(x)
        x_hat = self.decoder(hidden)
        return x_hat


def _add_noise(x_tr, noise_factor):
    if noise_factor > 0:
        x_tr_in = x_tr + noise_factor*x_tr.std()*torch.randn_like(x_tr)
        x_tr_target = x_tr
    else:
        x_tr_in = x_tr_target = x_tr
    return x_tr_in, x_tr_target


def train_model(
    device="cuda",
    batch_size=128,
    num_epochs=250,
    log_interval=250,
    image_dim=28**2,
    code_size=32,
    noise_factor=0.5,
    num_valid_batches=10,
    encoder_layer_sizes=None,
    decoder_layer_sizes=None
):
    encoder_layer_sizes = encoder_layer_sizes or (image_dim, 1024, code_size)
    decoder_layer_sizes = decoder_layer_sizes or (code_size, 256)

    ae = AutoEncoder(
        device=device,
        encoder_layer_sizes=encoder_layer_sizes,
        decoder_layer_sizes=decoder_layer_sizes,
        output_size=image_dim
    )

    optimizer = torch.optim.Adam(
        lr=1e-3,
        params=ae.parameters()
    )

    loss_fcn = nn.MSELoss()
    step = 0
    rloss_tr = deque(maxlen=log_interval)
    for epoch in range(num_epochs):
        bg_train = batch_generator(istrain=True, batch_size=batch_size)
        for x_tr, _ in bg_train:
            optimizer.zero_grad()
            x_tr = x_tr.to(device)
            x_tr_in, x_tr_target = _add_noise(x_tr, noise_factor)
            x_tr_hat = ae(x_tr_in)
            loss = loss_fcn(x_tr_target, x_tr_hat).mean()
            loss.backward()
            optimizer.step()

            rloss_tr.append(loss.item())

            with torch.no_grad():
                if step % log_interval == 0:
                    bg_val = batch_generator(istrain=False, batch_size=batch_size)
                    rloss_val = []
                    for i, (x_val, _) in enumerate(bg_val):
                        if i >= num_valid_batches:
                            break
                        x_val = x_val.to(device)
                        x_val_in, x_val_target = _add_noise(x_val, noise_factor)
                        x_val_hat = ae(x_val_in)
                        loss = loss_fcn(x_val_target, x_val_hat)
                        rloss_val.append(loss.item())

                    loss_log_tr = np.mean(rloss_tr)
                    log_loss_val = np.mean(rloss_val)
                    msg = f"epoch: {epoch}   step:  {step}  loss-tr: {loss_log_tr:0.3f}  " + \
                        f"loss-val: {log_loss_val:0.3f}"
                    print(msg)
                    _, ax = plt.subplots(nrows=1, ncols=3)
                    ax[0].imshow(x_val_target[0].cpu().numpy().reshape(28, 28))
                    ax[1].imshow(x_val_in[0].cpu().numpy().reshape(28, 28))
                    ax[2].imshow(x_val_hat[0].cpu().numpy().reshape(28, 28))
                    plt.savefig(join(data_dir, "img.png"))
                    plt.close()

            step += 1
    torch.save(ae, model_path)


def compute_clusters():
    batch_size = 512
    model = torch.load(model_path)
    bg = batch_generator(istrain=False, batch_size=batch_size)
    code = []
    targets = []
    for x, y in bg:
        x = x.squeeze().reshape(batch_size, -1).to(model.device)
        with torch.no_grad():
            code.append(model.encoder(x).cpu().numpy())
        targets.append(y.cpu().numpy())
    code = np.concatenate(code, axis=0)
    targets = np.concatenate(targets)
    return code, targets


def plot_clusters(code, targets):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = [
        "red", "darkorange", "lime", "darkgreen", "deepskyblue",
        "blue", "darkorchid", "plum", "dimgrey", "gold"
    ]

    tsne = TSNE()
    code = tsne.fit_transform(code)

    for digit, color in zip(sorted(np.unique(targets)), colors):
        idx = targets == digit
        ax.scatter(code[idx, 0], code[idx, 1], c=color, label=digit, s=10., alpha=0.5)

    ax.legend()
    plt.savefig(join(data_dir, "cluster.png"))
    plt.close()


def main():
    train_model()
    code, targets = compute_clusters()
    plot_clusters(code=code, targets=targets)
