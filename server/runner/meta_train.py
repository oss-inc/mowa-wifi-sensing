import os
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as DATA
import torch.nn.functional as F
import numpy as np
from dataloader.dataset import FSLDataset
from runner.utils import torch_seed, get_config, extract_train_sample
from torch.optim import lr_scheduler, Adam
import runner.proto as proto

class Trainer_FSL:
    def __init__(self, config):
        self.config = get_config(config)
        self.use_cuda = self.config['GPU']['cuda']
        self.device_ids = self.config['GPU']['gpu_ids']
        self.win_size = self.config['FSL']['dataset']['window_size']
        self.epochs = self.config['FSL']['train']['epoch']

        self.net = proto.load_protonet_vit(
            in_channels=self.config["model"]["ViT"]["in_channels"],
            patch_size=(self.config["model"]["ViT"]["patch_size"], self.config['subcarrier'][self.config['FSL']['dataset']["bandwidth"]]),
            embed_dim=self.config["model"]["ViT"]["embed_dim"],
            num_layers=self.config["model"]["ViT"]["num_layers"],
            num_heads=self.config["model"]["ViT"]["num_heads"],
            mlp_dim=self.config["model"]["ViT"]["mlp_dim"],
            # num_classes=len(self.config['FSL']['dataset']["train_activity_labels"]),
            num_classes=5,

            in_size=[self.config['FSL']['dataset']["window_size"], self.config['subcarrier'][self.config['FSL']['dataset']["bandwidth"]]]
        )
    
        self.optimizer = Adam(self.net.parameters(), lr=self.config['FSL']['train']['lr'])
        self.loss = nn.CrossEntropyLoss()

        self.way = self.config['FSL']['train']['n_way']
        self.support = self.config['FSL']['train']['n_support']
        self.query = self.config['FSL']['train']['n_query']

        if self.use_cuda:
            self.net.to(self.device_ids[0])
            self.loss.to(self.device_ids[0])
    
    def train(self) :
        torch_seed(40)
        print("Cuda: ", torch.cuda.is_available())
        print("Device id: ", self.device_ids[0])
        print(f"Load Train Dataset.. # window_size:{self.win_size}")

        train_data = FSLDataset(self.config['FSL']['dataset']['train_dataset_path'],
                                win_size=self.win_size,
                                mode='train',
                                )

        scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.config['FSL']['train']['step_size'], gamma=self.config['FSL']['train']['gamma'])

        epoch = 0
        epoch_size = self.config['FSL']['train']["epoch_size"]

        data_x, data_y = train_data.data_x, train_data.data_y
        data_x = np.expand_dims(data_x, axis=1)
        
        best_model = None
        best_accuracy = 0.0
        
        for epoch in range(self.epochs) :
            running_loss = 0.0
            running_acc = 0.0
            
            # Episode: epoch
            for episode in tqdm.tqdm(range(epoch_size), desc="Epoch {:d} train".format(epoch + 1)):
                sample = extract_train_sample(self.way, self.support, self.query, data_x, data_y)
                self.optimizer.zero_grad()
                loss, output = self.net.proto_train(sample)
                running_loss += output['loss']
                running_acc += output['acc']
                loss.backward()
                self.optimizer.step()
                
            epoch_loss = running_loss / epoch_size
            epoch_acc = running_acc / epoch_size

            print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, epoch_loss, epoch_acc))

            epoch += 1
            scheduler.step()

            if epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model = self.net.state_dict()    

            os.makedirs(self.config['FSL']['train']['save_path'], exist_ok=True)
            torch.save(self.net.state_dict(), os.path.join(self.config['FSL']['train']['save_path'], "{}.tar".format(epoch)))
            print("saved at {}".format(os.path.join(self.config['FSL']['train']['save_path'], "{}.tar".format(epoch))))

        if best_model is not None:
            os.makedirs(self.config['FSL']['train']['save_path'], exist_ok=True)
            torch.save(best_model, os.path.join(self.config['FSL']['train']['save_path'], "fsl_best_model.pt"))
            print("Best model saved at", os.path.join(self.config['FSL']['train']['save_path'], "fsl_best_model.pt"))