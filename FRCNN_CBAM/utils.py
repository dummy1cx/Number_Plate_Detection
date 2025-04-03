import torch
from torch.utils.data import DataLoader, random_split

def split_data(dataset, batch_size):
    total_size = len(dataset)
    train_size = int(0.75 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def print_model_size(model):
    total_pars = 0
    for _n, _par in model.state_dict().items():
        total_pars += _par.numel()
    print(f"Total number of parameters: {total_pars}")
    return