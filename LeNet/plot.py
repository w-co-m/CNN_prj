from torchvision.datasets import FashionMNIST
import numpy as np
from torchvision import transforms
import torch.utils.data as Data

train_data=FashionMNIST(root='./data',
                        train=True, transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
                        download=True)

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=100,
                               shuffle=True,
                               num_workers=0)

#获取一个Batch的数据
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
    batch_x = b_x.squeeze().numpy()
    batch_y=b_y.numpy()
    class_label = train_data.classes
    print(class_label)

