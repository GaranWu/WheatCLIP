import argparse
import random
import time
import os

import torch.optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import math
import dataset
from  model import *
from ContrastiveLoss import *


workers = 1
def main():
    print("")
    parser = argparse.ArgumentParser('Set parameters for training ', add_help=False)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--dataset', default="/home/liyaoxi/data/gwhd/gwhd_2021/", type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--img_size", default=512, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--layer_num", default=4, type=int)
    parser.add_argument("--backbone_name", default='vgg16', type=str)
    parser.add_argument("--delta", default=1, type=int)
    parser.add_argument("--output_dir", type=str, default="out", help="Output directory")
    parser.add_argument("--seq_all", default=336, type=int)
    parser.add_argument("--random_select",  action="store_false", help="是否启用随机选择样本处理（默认为 True）")
    parser.add_argument("--boxHeight", default=100, type=int)
    parser.add_argument("--boxWidth", default=100, type=int)
    parser.add_argument("--crop_size", default=50, type=int)
    parser.add_argument("--alpha", default=0.01, type=float)
    parser.add_argument("--number", default=2, type=int)
    parser.add_argument("--negative_num", default=10, type=int)
    parser.add_argument("--negative",  action="store_false", help="是否启用负样本处理（默认为 True）")
    args = parser.parse_args()




    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    # 定义训练参数
    device = args.device
    learning_rate = args.lr
    alpha = args.alpha
    # 网络超参数
    batch_size = args.batch_size
    epoch = args.epoch
    img_size = args.img_size
    save_file = os.path.join(args.output_dir, "best.pth")

    train_root = args.dataset + 'train/'
    train_ann = args.dataset + 'annotations/train_without_zero.csv'
    train_density = args.dataset + 'annotations/train_64_density'

    val_root = args.dataset + 'val/'
    val_ann = args.dataset + 'annotations/val_without_zero.csv'
    val_density = args.dataset + 'annotations/val_64_density'

    test_root = args.dataset + 'test/'
    test_ann = args.dataset + 'annotations/test_without_zero.csv'
    test_density = args.dataset + 'annotations/test_64_density'

    train_json = args.dataset + 'annotations/train_clip.json'
    train_negative_json = args.dataset + 'annotations/train_negative.json'
    val_json = args.dataset + 'annotations/val_clip.json'
    test_json = args.dataset + 'annotations/test_yolo_clip.json'
    print(train_ann, train_density, val_ann, val_density, test_ann, test_density,train_negative_json)
    # 准备数据集
    train_dataset = dataset.Countgwhd(img_path=train_root, ann_path=train_ann, density_path=train_density, clip_json= train_json,resize_shape=img_size,
                                      random_select=args.random_select, boxHeight=args.boxHeight, boxWidth=args.boxWidth,negative=args.negative,negative_json=train_negative_json,
                                      number=args.number, negative_num=args.negative_num, crop_size=args.crop_size)
    val_dataset = dataset.Countgwhd(img_path=val_root, ann_path=val_ann,  density_path=val_density,clip_json= val_json,resize_shape=img_size,
                                    random_select=args.random_select, boxHeight=args.boxHeight, boxWidth=args.boxWidth, number=args.number, crop_size=args.crop_size)
    test_dataset = dataset.Countgwhd(img_path=test_root, ann_path=test_ann, density_path=test_density,clip_json= test_json, resize_shape=img_size,
                                     random_select=args.random_select, boxHeight=args.boxHeight, boxWidth=args.boxWidth, number=args.number, crop_size=args.crop_size)

    # # seed = random.randint(1,100000)
    # seed = 77777
    # print(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

    #创建网络模型
    model = Multi_Granularity(layer_num=args.layer_num, device=device)
    model.to(device)
    print(model)
    #创建损失函数
    loss_fn = nn.MSELoss(reduction="sum")
    loss_fn = loss_fn.to(device)
    loss_infoNce = ContrastiveLoss()
    loss_infoNce = loss_infoNce.to(device)
    print(loss_fn)
    print(loss_infoNce.temperature)

    #优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epoch*0.8)], gamma=0.1)
    print(optimizer)


    torch.set_num_threads(workers)
    # writer = SummaryWriter(os.path.join(args.output_dir, "log"))
    best = 1000
    val_epoch = 0
    #训练
    for i in range(epoch):
        start1 = time.time()
        print("----------epoch: {}, lr: {}----------".format(i + 1, optimizer.param_groups[0]['lr']))
        loss_one_epoch = train(train_dataset, model, loss_fn,loss_infoNce, optimizer, lr_scheduler, batch_size, alpha, device,args.negative)
        # writer.add_scalar("train_loss", loss_one_epoch, i)
        end1 = time.time()
        print("这轮所用时间为：{}min \n\n".format((end1-start1)/60))

        if (i+1) % 2 == 0 :
            val_epoch = val_epoch + 1
            start2 = time.time()
            print("----------开始验证----------")
            prec = val(val_dataset, model, device)
            if prec < best:
                best = prec
                torch.save(model.state_dict(), save_file)
            end2 = time.time()
            print("测试所用时间为：{}min".format((end2-start2)/60))
            print("当前最好的mae为：{}".format(best))
            # writer.add_scalar("val_mae", prec, val_epoch)

    print("\n----------开始测试----------")
    test(test_dataset, model, device, best,save_file)
    # writer.close()

def train(train_dataset, model, loss_fn,loss_infoNce, optimizer, lr_scheduler, batch_size, alpha,device,negative):
    # 加载数据集
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                  num_workers=workers)
    model.train()
    model.negative = negative
    loss_ave = 0
    count_loss_avg = 0
    contrast_loss_avg = 0
    print_freq = 20
    data_num = 0
    for data in train_dataloader:

        data_num = data_num + 1
        imgs, targets, crop_images = data


        imgs = imgs.to(device)
        targets = targets.to(device).float()
        crop_images = crop_images.to(device)

        if model.negative:
            output, contrast_feature = model(imgs, crop_images)
        else:
            output = model(imgs, crop_images)

        count_loss = loss_fn(output, targets)

        if model.negative:
            info_nce_loss = loss_infoNce(contrast_feature)
            loss = count_loss + alpha*info_nce_loss
        else:
            loss = count_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_ave = loss_ave + loss.item()
        count_loss_avg = count_loss_avg + count_loss.item()
        if model.negative:
            contrast_loss_avg = contrast_loss_avg + alpha*info_nce_loss.item()

        if data_num % print_freq == 0:
            if model.negative:
                print("---loss: {}, count loss:{}, contrast loss:{}---".format(loss.item(), count_loss.item(), alpha*info_nce_loss.item()))
            else:
                print("---loss: {}, count loss:{}, ".format(loss.item(), count_loss.item()))
    if model.negative:
        print("----------本轮的平均loss为：{}, count loss:{}, contrast loss:{}  ----------\n".format(loss_ave / data_num,
                                                                                                count_loss_avg/data_num,
                                                                                                    contrast_loss_avg / data_num))
    else:
        print("----------本轮的平均loss为：{}, count loss:{}  ----------\n".format(loss_ave / data_num,
                                                                                                count_loss_avg/data_num))
    lr_scheduler.step()
    return loss_ave / data_num

def val(val_dataset, model, device):
    #加载数据集
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    model.eval()
    model.negative = False

    mae = 0.0
    mse = 0.0
    i = 0

    for data in val_dataloader:
        i = i + 1
        imgs, targets, crop_images = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        crop_images = crop_images.to(device)
        with torch.no_grad():
            output = model(imgs, crop_images)
            count = torch.sum(output).item()

        gt_count = torch.sum(targets).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)

        if i % 15 == 0:
            print("真实数量：{}     \t 预测数量：{}".format(gt_count, count))

    mae = mae * 1.0 / i
    mse = math.sqrt(mse / i)
    print("此次测试结果为：MAE：{}  \t MSE：{}".format(mae, mse))

    return mae

def test(test_dataset, model, device, best, save_file):
    #加载数据集
    if best < 1000:
        model.load_state_dict(torch.load(save_file))
    else:
        pass
    val_dataloader = DataLoader(test_dataset, batch_size=1)
    model.eval()
    model.negative = False

    mae = 0.0
    mse = 0.0
    i = 0
    for data in val_dataloader:
        i = i + 1
        imgs, targets, crop_images = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        crop_images = crop_images.to(device)
        with torch.no_grad():
            output = model(imgs, crop_images)
            count = torch.sum(output).item()

        gt_count = torch.sum(targets).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)

        print("真实数量：{}     \t 预测数量：{}".format(gt_count, count))

    mae = mae * 1.0 / i
    mse = math.sqrt(mse / i)
    print("此次测试结果为：MAE：{}  \t MSE：{}".format(mae, mse))


if __name__ == '__main__':
    main()







