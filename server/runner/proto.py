import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from model.vit import ViT

from runner.utils import euclidean_dist


class ProtoNet(nn.Module):
    def __init__(self, encoder):
        """
        Args:
            encoder : CNN encoding the dataloader dataframes in sample
            n_way (int): number of classes in a classification task
            n_support (int): number of labeled examples per class in the support set
            n_query (int): number of labeled examples per class in the query set
        """
        super(ProtoNet, self).__init__()
        self.encoder = encoder.cuda(0)

    def proto_train(self, sample):
        
        """
            Computes loss, accuracy and output for classification task
                Args:
                    sample (torch.Tensor): shape (n_way, n_support+n_query, (dim))
            Returns:
                torch.Tensor: shape(2), loss, accuracy and y_hat
        """

        sample_images = sample['csi_mats'].cuda(0)
        n_way = sample['n_way']
        n_support = sample['n_support']
        n_query = sample['n_query']

        x_support = sample_images[:, :n_support]
        x_query = sample_images[:, n_support:]

        # target indices are 0 ... n_way-1
        target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        target_inds = target_inds.cuda(0)

        # encode dataloader dataframes of the support and the query set
        '''
        Modified
        # Separate support and query tensor
        '''
        x_support = x_support.contiguous().view(n_way * n_support, *x_support.size()[2:])
        x_query = x_query.contiguous().view(n_way * n_query, *x_query.size()[2:])

        z_support = self.encoder.forward(x_support)
        z_query = self.encoder.forward(x_query)
        z_support_dim = z_support.size(-1)
        z_proto = z_support.view(n_way, n_support, z_support_dim).mean(1)

        # compute distances
        dists = euclidean_dist(z_query, z_proto)

        # compute probabilities
        log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()  # y_hat과 gt 같은지 비교
      
        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'y_hat': y_hat
        }

    def create_protoNet(self, support_sample):
        """
        Computes loss, accuracy and output for classification task
        Args:
            sample (torch.Tensor): shape (n_way, n_support+n_query, (dim))
        Returns:
            torch.Tensor: shape(2), loss, accuracy and y_hat
        """
        sample_images = support_sample['s_csi_mats'].cuda(0)
        n_way = support_sample['n_way']
        n_support = support_sample['n_support']

        x_support = sample_images

        # print(sample_images)
        # print(n_way)
        # print(n_support)
        # print(x_support.shape)

        # encode dataloader dataframes of the support and the query set
        '''
        Modified
        # Separate support and query tensor
        '''

        x_support = x_support.contiguous().view(n_way * n_support, *x_support.size()[2:])
        z_support = self.encoder.forward(x_support)
        z_support_dim = z_support.size(-1)
        z_proto = z_support.view(n_way, n_support, z_support_dim).mean(1)

        return z_proto

    def proto_test(self, query_sample, z_proto, n_way, gt):
        sample_images = query_sample.unsqueeze(0).cuda(0)
        n_query = 1

        #gt_mat = torch.tensor([gt] * n_way).cuda(0)

        x_query = sample_images.contiguous().view(*sample_images.size())
        z_query = self.encoder.forward(x_query)
        # compute distances
        dists = euclidean_dist(z_query, z_proto)

        # compute probabilities
        log_p_y = F.log_softmax(-dists, dim=1)
        y_hat = np.argmax(log_p_y[0].cpu())
        
        acc = 0
        # y_hat과 gt 같은지 비교
        if y_hat == gt:
            acc = 1

        print('label:{}, correct:{}'.format(gt, acc))

        return {
            'acc': acc,
            'y_hat': y_hat
            # ,'target':target
        }

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def load_protonet_vit(in_channels, patch_size, embed_dim, num_layers, num_heads, mlp_dim, num_classes, in_size):
    encoder = ViT(
        in_channels=in_channels,  # 입력 채널 수
        patch_size=patch_size,  # 패치 크기 (세로, 가로) 242 = 2 * 11 * 11
        embed_dim=embed_dim,  # 임베딩 차원
        num_layers=num_layers,  # Transformer 블록 수
        num_heads=num_heads,  # 멀티헤드 어텐션에서의 헤드 수
        mlp_dim=mlp_dim,  # MLP의 확장 비율
        num_classes=num_classes,  # 분류할 클래스 수
        in_size=in_size  # 입력 이미지 크기 (가로, 세로)
    )

    return ProtoNet(encoder)
