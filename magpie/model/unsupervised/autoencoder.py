from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F

from magpie import model


class Autoencoder(model.MagpieModel, prefix="ae"):

    def __init__(self, config):
        super().__init__(config)
        self.hidden_sizes = config["hidden_sizes"]
        self.input_size = config["input_size"]
        self.tied_type = config["tied_weights"]
        self._init()

    def _init(self):
        enc_sizes = [self.input_size] + self.hidden_sizes
        weight_pairs = list(zip(enc_sizes[:-1], enc_sizes[1:]))
        self.enc_weights = [nn.Parameter(model.make_ortho_weight(x[0], x[1])) for x in weight_pairs]
        self.enc_biases = [nn.Parameter(torch.zeros(x[1])) for x in weight_pairs]
        self._enc_weights = nn.ParameterList(self.enc_weights)
        self._enc_biases = nn.ParameterList(self.enc_biases)

        if self.tied_type == "none":
            self.dec_weights = [nn.Parameter(model.make_ortho_weight(x[0], x[1])) for x in weight_pairs]
        elif self.tied_type == "equal":
            self.dec_weights = self.enc_weights
        self.dec_biases = [nn.Parameter(torch.zeros(x[0])) for x in weight_pairs]
        self._dec_weights = nn.ParameterList(self.dec_weights)
        self._dec_biases = nn.ParameterList(self.dec_biases)

    def encode(self, x):
        for idx, (enc_weight, enc_bias) in enumerate(zip(self.enc_weights, self.enc_biases)):
            x = F.linear(x, enc_weight, bias=enc_bias)
            if idx != len(self.enc_weights) - 1:
                x = F.tanh(x)
        return x

    def decode(self, x):
        dec_params = list(zip(self.dec_weights, self.dec_biases))
        for idx, (dec_weight, dec_bias) in enumerate(reversed(dec_params)):
            x = F.linear(x, dec_weight.t(), bias=dec_bias)
            if idx != len(self.enc_weights) - 1:
                x = F.tanh(x)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


class ContractiveAutoencoder(model.MagpieModel, prefix="cae"):

    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config["hidden_size"]
        self.input_size = config["input_size"]
        self.tied_type = config["tied_weights"]
        self._init()

    def _init(self):
        self.enc_weight1 = nn.Parameter(model.make_ortho_weight(self.input_size, self.input_size * 3))
        self.enc_weight2 = nn.Parameter(model.make_ortho_weight(self.input_size * 3, self.hidden_size))
        self.enc_bias1 = nn.Parameter(torch.zeros(self.input_size * 3))
        self.enc_bias2 = nn.Parameter(torch.zeros(self.hidden_size))

        if self.tied_type == "equal":
            self.dec_weight1 = self.enc_weight1
            self.dec_weight2 = self.enc_weight2
        else:
            self.dec_weight1 = nn.Parameter(model.make_ortho_weight(self.input_size, self.input_size * 3))
            self.dec_weight2 = nn.Parameter(model.make_ortho_weight(self.input_size * 3, self.hidden_size))
        self.dec_bias1 = nn.Parameter(torch.zeros(self.input_size))
        self.dec_bias2 = nn.Parameter(torch.zeros(self.input_size * 3))

    def compute_contractive_loss(self, x):
        x = (1 - F.tanh(F.linear(x, self.enc_weight1, bias=self.enc_bias1))**2)
        x = x.unsqueeze(-1).expand(-1, -1, self.enc_weight1.size(1))
        J = self.enc_weight1 * x
        frob_norm = J.view(x.size(0), -1).norm(dim=1, p=2).mean()
        return frob_norm

    def encode(self, x):
        x = F.linear(x, self.enc_weight1, bias=self.enc_bias1)
        x = F.tanh(x)
        x = F.linear(x, self.enc_weight2, bias=self.enc_bias2)
        return x

    def decode(self, x):
        x = F.linear(x, self.dec_weight2.t(), bias=self.dec_bias2)
        x = F.tanh(x)
        x = F.linear(x, self.dec_weight1.t(), bias=self.dec_bias1)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


class ContractiveAutoencoderTrainer(model.MagpieTrainer, name="cae-trainer"):

    def __init__(self, config):
        super().__init__(config)
        self.loss_fn = nn.MSELoss()

    def loss(self, stage, model, input_tuple):
        x_in, x_out = input_tuple
        if self.config["use_cuda"]:
            x_in = x_in.cuda()
            x_out = x_out.cuda()
        x_recon = model(x_in)
        loss = self.loss_fn(x_recon, x_out)
        if stage == "train":
            loss = self.config["cae_decay"] * model.compute_contractive_loss(x_in) + loss
        return loss, x_recon


class AutoencoderTrainer(model.MagpieTrainer, name="ae-trainer"):

    def __init__(self, config):
        super().__init__(config)
        self.loss_fn = nn.MSELoss()

    def loss(self, stage, model, input_tuple):
        x_in, x_out = input_tuple
        if self.config["use_cuda"]:
            x_in = x_in.cuda()
            x_out = x_out.cuda()
        # if stage == "training":
        #     x_in += x_in.clone().normal_(0, 0.1)
        x_recon = model(x_in)
        loss = self.loss_fn(x_recon, x_out)
        return loss, x_recon
