import torch
import yaml
import random
import logging
import numpy as np

# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def euclidean_dist(x, y):
    """
    Computes euclidean distance btw x and y
    Args:
        x (torch.Tensor): shape (n, d). n usually n_way*n_query
        y (torch.Tensor): shape (m, d). m usually n_way
    Returns:
        torch.Tensor: shape(n, m). For each query, the distances to each centroid
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def extract_train_sample(n_way, n_support, n_query, datax, datay):

    """
    Picks random sample of size n_support+n_querry, for n_way classes
    Args:
        n_way (int): number of classes in a classification task
        n_support (int): number of labeled examples per class in the support set
        n_query (int): number of labeled examples per class in the query set
        datax (np.array): dataset of dataloader dataframes
        datay (np.array): dataset of labels
    Returns:
        (dict) of:
          (torch.Tensor): sample of dataloader dataframes. Size (n_way, n_support+n_query, (dim))
          (int): n_way
          (int): n_support
          (int): n_query
    """
    sample = None
    K = np.random.choice(np.unique(datay), n_way, replace=False)
    
    # print(datax.shape)
    # print(datay.shape)

    for cls in K:
        datax_cls = datax[datay == cls]
        perm = np.random.permutation(datax_cls)
        sample_cls = perm[:(n_support + n_query)]
        # print(datax_cls.shape)
        if sample is None:
            sample = np.array([sample_cls])
            # print(sample.shape)
            # print(sample_cls.shape)
        else:
            sample = np.vstack([sample, [np.array(sample_cls)]])
            # print(sample.shape)
            # print(sample_cls.shape)
        #sample.append(sample_cls)

    sample = np.array(sample)
    sample = torch.from_numpy(sample).float()

    # sample = sample.permute(0,1,4,2,3)
    # sample = np.expand_dims(sample, axis= 0)

    return ({
        'csi_mats': sample,
        'n_way': n_way,
        'n_support': n_support,
        'n_query': n_query
    })

def extract_test_sample(n_way, n_support, n_query, datax, datay, config):
    """
    Picks random sample of size n_support+n_querry, for n_way classes
    Args:
        n_way (int): number of classes in a classification task
        n_support (int): number of labeled examples per class in the support set
        n_query (int): number of labeled examples per class in the query set
        datax (np.array): dataset of csi dataframes
        datay (np.array): dataset of labels
    Returns:
        (dict) of:
          (torch.Tensor): sample of csi dataframes. Size (n_way, n_support+n_query, (dim))
          (int): n_way
          (int): n_support
          (int): n_query
    """
    #K = np.array(['empty', 'jump', 'stand', 'walk']) # ReWis
    K = np.array(config['FSL']['dataset']["test_activity_labels"])
    label_to_int = {label: index for index, label in enumerate(K)}
    K = np.array([label_to_int[label] for label in K])

    # extract support set & query set
    support_sample = []
    query_sample = []
    for cls in K:
        datax_cls = datax[datay == cls]
        # print(datax_cls.shape)
        # print(datax_cls.dtype)
        # print(datax_cls)

        support_cls = datax_cls[:n_support]
        query_cls = np.array(datax_cls[n_support:n_support+n_query])

        # print(query_cls.shape)
        # print(query_cls.dtype)
        # print("---------")

        support_sample.append(support_cls)
        query_sample.append(query_cls)
    
    support_sample = np.array(support_sample)
    query_sample = np.array(query_sample)

    # print(support_sample.dtype)
    # print(type(support_sample))

    # print(query_sample.dtype)
    # print(type(query_sample))

    support_sample = torch.from_numpy(support_sample).float()
    query_sample = torch.from_numpy(query_sample).float()

    # print("Utils.py")
    # print(support_sample.shape)
    # print(query_sample.shape)

    return ({
        's_csi_mats': support_sample,
        'q_csi_mats': query_sample,
        'n_way': n_way,
        'n_support': n_support,
        'n_query': n_query
    })


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: yellow + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    