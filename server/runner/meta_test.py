import tqdm
import torch
import torch.nn as nn
import torch.utils.data as DATA
import torch.nn.functional as F
import numpy as np
from dataloader.dataset import FSLDataset
from runner.utils import torch_seed, get_config,extract_test_sample
import runner.proto as proto
from tqdm import tqdm


class Tester_FSL:
    def __init__(self, config):
        self.config = get_config(config)
        self.use_cuda = self.config['GPU']['cuda']
        self.device_ids = self.config['GPU']['gpu_ids']
        self.win_size = self.config['FSL']['dataset']['window_size']

        self.net = proto.load_protonet_vit(
            in_channels=self.config["model"]["ViT"]["in_channels"],
            patch_size=(self.config["model"]["ViT"]["patch_size"], self.config['subcarrier'][self.config['FSL']['dataset']["bandwidth"]]),
            embed_dim=self.config["model"]["ViT"]["embed_dim"],
            num_layers=self.config["model"]["ViT"]["num_layers"],
            num_heads=self.config["model"]["ViT"]["num_heads"],
            mlp_dim=self.config["model"]["ViT"]["mlp_dim"],
            # num_classes=len(self.config['FSL']['dataset']["test_activity_labels"]),
            num_classes=5,
            in_size=[self.config['FSL']['dataset']["window_size"], self.config['subcarrier'][self.config['FSL']['dataset']["bandwidth"]]]
        )
    
        self.loss = nn.CrossEntropyLoss()
        self.way = self.config['FSL']['test']['n_way']
        self.support = self.config['FSL']['test']['n_support']
        self.query = self.config['FSL']['test']['n_query']

        if self.use_cuda:
            self.net.to(self.device_ids[0])

    def test(self):
        torch_seed(40)
        print("Cuda: ", torch.cuda.is_available())
        print("Device id: ", self.device_ids[0])

        print(f"Load Test Dataset.. # window_size:{self.win_size}")
        test_data = FSLDataset(self.config['FSL']['dataset']['test_dataset_path'],
                                win_size=self.win_size,
                                mode='test', 
                                mac=False, time=False
                                )

        test_x, test_y = test_data.data_x, test_data.data_y
        test_x = np.expand_dims(test_x, axis=1)

        self.net.load_state_dict(torch.load(self.config['FSL']['test']['save_model_path']))

        running_acc = 0.0
        total_correct_predictions = 0
        total_predictions = 0

        conf_mat = torch.zeros(self.way, self.way)

        sample = extract_test_sample(self.way, self.support, self.query, test_x, test_y, self.config)
        query_samples = sample['q_csi_mats']

        # Create target domain Prototype Network with support set(target domain)
        z_proto = self.net.create_protoNet(sample)
        total_count = 0

        self.net.eval()
        with torch.no_grad():
            for episode in tqdm(range(self.config['FSL']['test']['epoch_size']), desc="test"):
                for label, q_samples in enumerate(query_samples):
                    for i in range(0, self.query):
                        output = self.net.proto_test(q_samples[i], z_proto, self.way, label)
                        pred_label = output['y_hat'] # only one class result

                        conf_mat[label][pred_label] += 1

                        running_acc += output['acc']
                        total_count += 1
        print(conf_mat)
        if total_count == 0:
            avg_acc = 0
        else :
            avg_acc = running_acc / total_count
        print('Test results -- Acc: {:.5f}'.format(avg_acc))


