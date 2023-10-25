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
from torch.optim import lr_scheduler, Adam


class Trainer_SVL:
    def __init__(self, config):
        self.config = get_config(config)
        self.use_cuda = self.config['GPU']['cuda']
        self.device_ids = self.config['GPU']['gpu_ids']
        self.batch_size = self.config['SVL']['train']['batch_size']
        self.train_proportion = self.config['SVL']['dataset']['train_proportion']
        self.win_size = self.config['SVL']['dataset']['window_size']
        self.epochs = self.config['SVL']['train']['epoch']

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
        
        self.optimizer = Adam(self.net.parameters(), lr=self.config['SVL']['train']['lr'])
        self.loss = nn.CrossEntropyLoss()

        if self.use_cuda:
            self.net.to(self.device_ids[0])
            self.loss.to(self.device_ids[0])


    def train(self):
        # fix torch seed
        torch_seed(40)
        print("Cuda: ", torch.cuda.is_available())
        print("Device id: ", self.device_ids[0])

        print(f"Load Train Dataset.. # window_size:{self.win_size}")
        train_data = SVLDataset(self.config['SVL']['dataset']['dataset_path'],
                                win_size=self.win_size,
                                mode='train',
                                train_proportion=self.train_proportion)

        train_dataloader = DATA.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.config['SVL']['train']['step_size'], gamma=self.config['SVL']['train']['gamma'])

        best_model = None
        best_accuracy = 0.0
        
        self.net.train()
        for epoch in range(self.epochs):
            print('Epoch {}/{}'.format(epoch + 1, self.epochs))
            print('-' * 10)

            train_loss = 0.0
            train_acc = 0.0
            total_iter = 0

            for i, data in enumerate(tqdm.tqdm(train_dataloader)):
                data_x, data_y = data
                data_x = data_x.unsqueeze(1).float()
                data_y = data_y.long()

                if self.use_cuda:
                    data_x = data_x.to(self.device_ids[0])
                    data_y = data_y.to(self.device_ids[0])

                self.optimizer.zero_grad()

                outputs = self.net(data_x)
                loss = self.loss(outputs, data_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                
                # Calculate accuracy
                outputs = F.log_softmax(outputs, dim=1)
                y_hat = torch.from_numpy(np.array([np.argmax(outputs.cpu().data.numpy()[ii]) for ii in range(len(outputs))]))
                
                train_acc += torch.eq(y_hat, data_y.cpu()).float().mean()
                total_iter += 1

            epoch_loss = train_loss / total_iter
            epoch_acc = train_acc / total_iter
            print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, epoch_loss, epoch_acc))
            scheduler.step()

            if epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model = self.net.state_dict()    
            
            os.makedirs(self.config['SVL']['train']['save_path'], exist_ok=True)
            torch.save(self.net.state_dict(), os.path.join(self.config['SVL']['train']['save_path'], "{}.tar".format(epoch)))
            print("saved at {}".format(os.path.join(self.config['SVL']['train']['save_path'], "{}.tar".format(epoch))))
        
        if best_model is not None:
            os.makedirs(self.config['SVL']['train']['save_path'], exist_ok=True)
            torch.save(best_model, os.path.join(self.config['SVL']['train']['save_path'], "svl_best_model.pt"))
            print("Best model saved at", os.path.join(self.config['SVL']['train']['save_path'], "svl_best_model.pt"))