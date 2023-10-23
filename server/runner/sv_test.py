# SVL Test
import os
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as DATA
import torch.nn.functional as F
import numpy as np
from dataloader.dataset import SVLDataset
from model.vit import ViT
from runner.utils import torch_seed, get_config
from plot.conf_matrix import plot_confusion_matrix


class Tester_SVL:
    def __init__(self, config):
        self.config = get_config(config)
        self.use_cuda = self.config['GPU']['cuda']
        self.device_ids = self.config['GPU']['gpu_ids']
        self.batch_size = self.config['SVL']['train']['batch_size']
        self.train_proportion = self.config['SVL']['dataset']['train_proportion']
        self.activities = self.config['SVL']['dataset']['activity_labels']
        self.win_size = self.config['SVL']['dataset']['window_size']

        self.net = ViT(
            in_channels=self.config["model"]["ViT"]["in_channels"],
            patch_size=(self.config["model"]["ViT"]["patch_size"], self.config['subcarrier'][self.config['SVL']['dataset']["bandwidth"]]),
            embed_dim=self.config["model"]["ViT"]["embed_dim"],
            num_layers=self.config["model"]["ViT"]["num_layers"],
            num_heads=self.config["model"]["ViT"]["num_heads"],
            mlp_dim=self.config["model"]["ViT"]["mlp_dim"],
            num_classes=len(self.config['SVL']['dataset']["activity_labels"]),
            in_size=[self.config['SVL']['dataset']["window_size"], self.config['subcarrier'][self.config['SVL']['dataset']["bandwidth"]]]
            )

        self.loss = nn.CrossEntropyLoss()

        if self.use_cuda:
            self.net.to(self.device_ids[0])


    def test(self):
        # fix torch seed
        torch_seed(40)
        print("Cuda: ", torch.cuda.is_available())
        print("Device id: ", self.device_ids[0])

        print(f"Load Test Dataset.. # window_size:{self.win_size}")
        test_data = SVLDataset(self.config['SVL']['dataset']['dataset_path'],
                                win_size=self.win_size,
                                mode='test',
                                train_proportion=self.train_proportion)

        test_dataloader = DATA.DataLoader(test_data, batch_size=self.batch_size, shuffle=True)

        # Load trained model
        self.net.load_state_dict(torch.load(self.config['SVL']['test']['save_model_path']))

        test_acc = 0
        test_loss = 0
        total_iter = 0
        conf_mat = torch.zeros(len(self.activities), len(self.activities))

        self.net.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm.tqdm(test_dataloader)):
                data_x, data_y = data
                data_x = data_x.unsqueeze(1).float()
                data_y = data_y.long()

                if self.use_cuda:
                    data_x = data_x.to(self.device_ids[0])
                    data_y = data_y.to(self.device_ids[0])

                outputs = self.net(data_x)
                loss = self.loss(outputs, data_y)
                test_loss += loss.item()
                
                # Calculate accuracy
                outputs = F.log_softmax(outputs, dim=1)
                y_hat = torch.from_numpy(np.array([np.argmax(outputs.cpu().data.numpy()[ii]) for ii in range(len(outputs))]))
                test_acc += torch.eq(y_hat, data_y.cpu()).float().mean()
                total_iter += 1
            
            test_loss = test_loss / total_iter
            test_acc = test_acc / total_iter
            print('Test Result -- Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))