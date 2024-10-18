import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.JoinG import *
from model.resnets import resnet101

from utils.dataloader import *
from utils.utils import *
from utils.accuracy import *
from train import *
from test import *

np.set_printoptions(threshold=np.inf)

if __name__ == "__main__":
    classes_path = 'dataset/cls_classes (12).txt'
    cuda = True
    logs_dir = 'logs'
    dp = False
    input_shape = [200, 200]
    epoch = 10
    lr = 0.0001
    momentum = 0.9
    weight_decay = 5e-4
    batch_size = 16
    save_period = epoch // 2
    # logs_dir = 'logs'
    # checkpoints_dir = "checkpoints"
    train_annotation_path = "images/SIRI-WHU_images/cls_train.txt"
    test_annotation_path = "images/SIRI-WHU_images/cls_test.txt"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    time_now = time.localtime()
    logs_folder = os.path.join(logs_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time_now))
    os.makedirs(logs_folder)
    class_names, num_classes = get_classes(classes_path)


# =================================================================================#
    pretrained_dict = torch.load('pths/JoinG.pth')
    model =JoinG()  # 确保此处与您的模型架构一致
    # 读取参数
    model_dict = model.state_dict()
    # 将pretrained_dict里不属于model_dict的键剔除掉
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 更新现有的model_dict
    model_dict.update(pretrained_dict)
    # 加载我们真正需要的state_dict
    model_dict.pop('fc.weight')
    model_dict.pop('fc.bias')
    # 加载剩余的权重
    model.load_state_dict(pretrained_dict, strict=False)
    inchannel = model.fc.in_features
    model.fc = nn.Linear(inchannel, num_classes)
#=================================================================================#

    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(train_annotation_path, encoding="utf-8") as f:
        train_lines = f.readlines()
    with open(test_annotation_path, encoding="utf-8") as f:
        test_lines = f.readlines()
    num_train = len(train_lines)
    num_test = len(test_lines)
    #np.random.seed(3047)
    np.random.shuffle(train_lines)

    print("device:", device, "num_train:", num_train, "num_test:", num_test)
    print("===============================================================================")
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch//2, gamma=0.9)
    criterion = nn.CrossEntropyLoss().cuda()


    transform_train = transforms.Compose([
        transforms.RandomCrop(200),  # 先四周填充0，再把图像随机裁剪成128x128
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.RandomCrop(200),  # 先四周填充0，再把图像随机裁剪成128x128
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = MyDataset(train_lines, input_shape=input_shape, transform=transform_train)
    test_dataset = MyDataset(test_lines, input_shape=input_shape, transform=transform_test)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    #---------------------------------------#
    #   开始模型训练
    #---------------------------------------#
    print("start training")
    epoch_result = np.zeros([4, epoch])
    for e in range(epoch):
        model.train()
        train_acc1, train_acc3, train_loss = train_epoch(model, train_loader, criterion, optimizer, e, epoch, device)
        scheduler.step()
        #print("Epoch: {:03d} | train_loss: {:.4f} | train_acc1: {:.2f}% | train_acc3: {:.2f}%".format(e+1, train_loss, train_acc1, train_acc3))
        epoch_result[0][e], epoch_result[1][e], epoch_result[2][e], epoch_result[3][e]= e+1, train_loss, train_acc1, train_acc3
    print("save train logs successfully")
    draw_result_visualization(logs_folder, epoch_result)
    print("===============================================================================")

    print("start testing")
    model.eval()
    test_acc1, test_acc3, test_prediction, test_label = test_epoch(model, test_loader, device)
    unique_labels = set(test_label)
    unique_predictions = set(test_prediction)


    test_CM, test_weighted_recall, test_weighted_precision, test_weighted_f1 = output_metrics(test_label, test_prediction)
    #print("Test Result  =>  Accuracy: {:.2f}%| W-Recall: {:.4f} | W-Precision: {:.4f} | W-F1: {:.4f}".format(test_acc1, test_weighted_recall, test_weighted_precision, test_weighted_f1))
    store_result(logs_folder,test_acc3, test_weighted_recall, test_weighted_precision, test_weighted_f1, test_CM, epoch, batch_size, lr, weight_decay)
    print("save test result successfully")
    print("===============================================================================")
