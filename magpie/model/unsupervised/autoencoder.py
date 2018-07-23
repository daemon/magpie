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

        self.enc_weight1 = nn.Parameter(make_ortho_weight(self.input_size, self.input_size * 2))
        self.enc_weight2 = nn.Parameter(make_ortho_weight(self.input_size * 2, self.hidden_size))
        self.enc_bias1 = nn.Parameter(torch.zeros(self.input_size * 2))
        self.enc_bias2 = nn.Parameter(torch.zeros(self.hidden_size))

        if self.tied:
            self.dec_weight1 = self.enc_weight1
            self.dec_weight2 = self.enc_weight2
        else:
            self.dec_weight1 = nn.Parameter(make_ortho_weight(self.input_size, self.input_size * 2))
            self.dec_weight2 = nn.Parameter(make_ortho_weight(self.input_size * 2, self.hidden_size))
        self.dec_bias1 = nn.Parameter(torch.zeros(self.input_size * 2))
        self.dec_bias2 = nn.Parameter(torch.zeros(self.hidden_size))

    def compute_contractive_loss(self, x):
        x = x.unsqueeze(1).expand(-1, self.enc_weight1.size(0), -1)
        b = self.enc_bias1.unsqueeze(1).expand(-1, x.size(2))
        J = self.enc_weight1 * (1 - F.tanh(self.enc_weight1 * x + b)**2)
        frob_norm = J.view(x.size(0), -1).norm(dim=1, p=2).mean()
        return frob_norm

    def encode(self, x):
        x = F.linear(x, self.enc_weight1, bias=self.enc_bias1)
        x = F.tanh(x)
        x = F.linear(x, self.enc_weight2, bias=self.enc_bias2)
        return x

    def decode(self, x):
        x = F.linear(x - self.dec_bias2, self.dec_weight2.t())
        x = atanh(x)
        x = F.linear(x - self.dec_bias1, self.dec_weight1.t())
        return x

    def forward(self, x):
        return self.decode(self.encode(x))
