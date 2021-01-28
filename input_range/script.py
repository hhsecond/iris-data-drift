def data_drift(x):
    temp = x.squeeze()
    if temp[0] < 5:
        return torch.ones(1)
    else:
        return torch.zeros(1)
