from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F

from magpie import model


class ATanhFunction(Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        eps = 1E-7
        return (1 + x).clamp_(min=eps).log_().sub_((1 - x).clamp_(min=eps).log_()).mul_(0.5)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_variables
        eps = 1E-7
        return grad_output / (x.pow(2).mul_(-1).add_(1).clamp_(min=eps))

atanh = ATanhFunction.apply


class ContractiveAutoencoder(model.MagpieModel, prefix="cae"):

    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config["hidden_size"]
        self.input_size = config["input_size"]
        self.tied = config["tied_weights"]
        self._init()

    def _init(self):
        def make_ortho_weight(input_size, output_size):
            return nn.init.orthogonal_(torch.empty(output_size, input_size))

        self.enc_weight1 = nn.Parameter(make_ortho_weight(self.input_size, self.input_size * 3))
        self.enc_weight2 = nn.Parameter(make_ortho_weight(self.input_size * 3, self.hidden_size))
        self.enc_bias1 = nn.Parameter(torch.zeros(self.input_size * 3))
        self.enc_bias2 = nn.Parameter(torch.zeros(self.hidden_size))

        if self.tied:
            self.dec_weight1 = self.enc_weight1
            self.dec_weight2 = self.enc_weight2
        else:
            self.dec_weight1 = nn.Parameter(make_ortho_weight(self.input_size, self.input_size * 3))
            self.dec_weight2 = nn.Parameter(make_ortho_weight(self.input_size * 3, self.hidden_size))
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

    def print_stats(self, stage, step, input_tuple, outputs, loss):
        print(f"{stage} #{step}: loss {loss:.5}")
