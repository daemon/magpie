from sys import stderr
import os

from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data

from . import model as mod
import magpie.dataset as ds


registry = {}

class MagpieTrainer(object):

    def __init__(self, config):
        self.config = config

    def __init_subclass__(cls, name, **kwargs):
        registry[name] = cls

    def loss(self, model, input_tuple):
        raise NotImplementedError

    def print_stats(self, stage, step, input_tuple, outputs, loss):
        print(f"{stage} #{step}: loss {loss:.5}")

    def train(self):
        cfg = self.config
        print("Using config: ", cfg)
        ds_cls = ds.find_dataset(cfg["dataset_type"])
        data_dict = torch.load(cfg["dataset_file"])
        train_set, dev_set, test_set = ds_cls.splits(cfg, data_dict)
        train_loader = data.DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
        dev_loader = data.DataLoader(dev_set, batch_size=cfg["batch_size"], shuffle=False)
        test_loader = data.DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False)

        try:
            os.makedirs(os.path.dirname(cfg["output_file"]))
        except FileExistsError:
            pass

        model_cls = mod.find_model(cfg["model_type"])
        model = model_cls(cfg)
        if cfg["use_cuda"]:
            model = model.cuda()

        params = list(filter(lambda x: x.requires_grad, model.parameters()))
        schedule = cfg["schedule"].copy()
        schedule.append(None)
        sched_idx = 0
        optimizer = optim.SGD(params, lr=cfg["lr"][sched_idx], momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"])
        best_dev_loss = np.inf
        train_step = 1
        for epoch_idx in range(cfg["epochs"]):
            num_epoch = epoch_idx + 1
            if num_epoch == schedule[sched_idx]:
                sched_idx += 1
                optimizer = optim.SGD(params, lr=cfg["lr"][sched_idx], momentum=cfg["momentum"],
                    weight_decay=cfg["weight_decay"])

            model.train()
            t = tqdm(train_loader, file=stderr, total=len(train_loader))
            for input_tuple in t:
                optimizer.zero_grad()
                loss, outputs = self.loss("train", model, input_tuple)
                loss.backward()
                optimizer.step()
                loss = loss.item()
                loss_txt = f"Train #{num_epoch} loss: {loss:.4}"
                t.set_description(loss_txt)
                self.print_stats("train", train_step, input_tuple, outputs, loss)
                train_step += 1

            model.eval()
            dev_losses = []
            t = tqdm(enumerate(dev_loader), file=stderr, total=len(dev_loader))
            for idx, input_tuple in t:
                loss, outputs = self.loss("dev", model, input_tuple)
                loss = loss.item()
                self.print_stats("dev", idx, input_tuple, outputs, loss)
                dev_losses.append(loss)
                loss_txt = f"Dev #{num_epoch} loss: {loss:.4}"
                t.set_description(loss_txt)
            dev_loss = np.mean(dev_losses)
            print(f"Final dev loss: {dev_loss}")
            if dev_loss < best_dev_loss:
                print("Saving best model...")
                torch.save(model.state_dict(), cfg["output_file"])
                best_dev_loss = dev_loss

def find_trainer(name):
    return registry[name]