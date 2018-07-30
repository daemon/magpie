import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from magpie import model


class SimilarityEncoder(model.MagpieModel, prefix="sim-enc"):

    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config["hidden_size"]
        self.input_size = config["input_size"]
        self.n_layers = config["n_layers"]
        self.freq_cnn = FrequencyCNN(config)
        self.encoder = model.find_rnn(config["rnn_type"])(self.hidden_size, 
            self.hidden_size, self.n_layers)

    def forward(self, x, lengths):
        x = self.freq_cnn(x)
        x = rnn_utils.pack_padded_sequence(x, lengths)
        rnn_seq, _ = self.encoder(x)
        rnn_seq = rnn_utils.pad_packed_sequence(x).permute(1, 0, 2).contiguous()
        rnn_seq = F.avg_pool2d(rnn_seq, (2, 1))
        return rnn_seq


class SimilarityModel(model.MagpieModel, prefix="sim"):

    def __init__(self, config):
        super().__init__(config)
        self.encoder = SimilarityEncoder(config)

    def compute_sim(self, x_vecs, y_vecs, lengths_x, lengths_y, normalize=True):
        J = x_vecs.size(1)
        K = y_vecs.size(1)
        D = torch.empty(x_vecs.size(0), J + 1, K + 1)
        D[:, 0, :] = 0
        D[:, :, 0] = 0
        for j, x in enumerate(x_vecs.split(1, 1)):
            for k, y in enumerate(y_vecs.split(1, 1)):
                candidates = torch.stack([D[:, j, k], D[:, j + 1, k], D[:, j, k + 1]])
                m = x.unsqueeze(-2).matmul(y.unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1)
                m = 1 - F.sigmoid(m)
                D[:, j + 1, k + 1] = torch.min(candidates, 0)[0] + m
        batch_dist = 0
        for idx, (len_x, len_y) in enumerate(zip(lengths_x.tolist(), lengths_y.tolist())):
            dist = D[idx, len_x, len_y]
            if normalize:
                dist = dist / (len_x + len_y)
            batch_dist += dist
        return batch_dist.mean()

    def forward(self, x, lengths_x, y, lengths_y, z=None, lengths_z=None):
        old_len_x = lengths_x
        old_len_y = lengths_y
        old_len_z = lengths_z

        lengths_x, x_idx = torch.sort(lengths_x, descending=True)
        x_vecs = self.encoder(x[x_idx], lengths_x)[x_idx.sort(descending=True)[1]]
        lengths_y, y_idx = torch.sort(lengths_y, descending=True)
        y_vecs = self.encoder(y[y_idx], lengths_y)[y_idx.sort(descending=True)[1]]
        sim1 = self.compute_sim(x_vecs, y_vecs, old_len_x, old_len_y)
        if z is None:
            lengths_z, z_idx = torch.sort(lengths_z, descending=True)
            z_vecs = self.encoder(z[z_idx], lengths_z)[z_idx.sort(descending=True)[1]]
            sim2 = self.compute_sim(x_vecs, z_vecs, old_len_x, old_len_z)
            return sim1, sim2
        return sim1,

class FrequencyCNN(model.MagpieModel, prefix="freq-cnn"):

    def __init__(self, config):
        super().__init__(config)
        self.input_size = config["input_size"]
        self.padding = nn.ConstantPad2d((0, 0, 2, 0), -3.14)
        self.conv = nn.Conv2d(5, config["hidden_size"], (self.input_size * 3 // 8, 3))
        self.bn = nn.BatchNorm2d(config["hidden_size"])

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.size(0), 5, -1, x.size(2))
        x = self.bn(F.relu(self.conv(self.padding(x)), inplace=True))
        x = x.permute(3, 0, 1, 2).contiguous()
        return x.view(x.size(0), x.size(1), -1)
