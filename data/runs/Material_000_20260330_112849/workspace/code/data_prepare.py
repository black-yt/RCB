class RealisticCrystalDataset:
    def __init__(self, data_list=None, **kwargs):
        self.data_list = data_list or []
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
