import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import GoogLeNet, inception

def test_data_process():
    test_data = FashionMNIST(root='D:\\布匹项目\\GoogLeNet\\data',
                             train=False,
                             transform=transforms.Compose([transforms.Resize(size=227), transforms.ToTensor()]),
                             download=False)
    test_loader = Data.DataLoader(dataset=test_data,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=2)
    return test_loader

def test_model_process(model,test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # 初始化测试参数
    test_correct = 0
    test_num = 0
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()
            # 测试输出结果
            test_output = model(test_data_x)
            # 计算预测标签
            test_predict = torch.max(test_output, dim=1)[1]
            # 计算测试准确率
            test_correct += torch.sum((test_predict == test_data_y).float())
            test_num += test_data_y.size(0)
    test_acc = test_correct.double().item() / test_num
    print("准确率:", test_acc)


if __name__ == '__main__':
    test_dataloader = test_data_process()
    model = GoogLeNet(inception)
    model.load_state_dict(torch.load('D:\\布匹项目\\GoogLeNet\\best_model.pth'))
    test_model_process(model,test_dataloader)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval()
            test_output = model(test_data_x)
            test_predict = torch.max(test_output, dim=1)[1]
            result = test_predict.item()
            label = test_data_y.item()
            print("预测结果:", result, "真实标签:", label)






