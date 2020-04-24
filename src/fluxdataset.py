import torch

class FluxDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, device = "cpu"):
        self.device = torch.device(device)
        self.dataset = torch.tensor(dataframe.values).float()
        print('using device {}'.format(self.device))
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = torch.index_select(self.dataset, 0, torch.tensor([idx]))
        return row[:, :1][0][0].long().to(self.device), row[:, 1:][0].to(self.device)