import argparse
import time
import os

import torch.optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math
import dataset
from  model import *
from ContrastiveLoss import *
def main():
    print("without contrast loss")
    parser = argparse.ArgumentParser('Set parameters for training ', add_help=False)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--dataset', default="/home/liyaoxi/data/gwhd/gwhd_2021/", type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument("--epoch", default=1000, type=int)
    parser.add_argument("--img_size", default=512, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--layer_num", default=4, type=int)
    parser.add_argument("--backbone_name", default='vgg16', type=str)
    parser.add_argument("--delta", default=1, type=int)
    parser.add_argument("--output_dir", type=str, default="out", help="Output directory")
    parser.add_argument("--seq_all", default=336, type=int)
    parser.add_argument("--random_select", default=False, type=bool)
    parser.add_argument("--boxHight", default=100, type=int)
    parser.add_argument("--boxWidth", default=100, type=int)
    parser.add_argument("--alpha", default=0.01, type=float)
    parser.add_argument("--number", default=2, type=int)
    args = parser.parse_args()

    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    # 定义训练参数
    device = args.device
    img_size = args.img_size
    save_file = os.path.join("best_844040.pth")
    print(save_file)


    test_root = args.dataset + 'test/'
    test_ann = args.dataset + 'annotations/test_without_zero.csv'
    test_density = args.dataset + 'annotations/test_64_density'

    test_json = args.dataset + 'annotations/test_yolo_clip.json'

    test_dataset = dataset.Countgwhd(img_path=test_root, ann_path=test_ann, density_path=test_density, clip_json=test_json,
                                 resize_shape=img_size,number=args.number,
                                 random_select=args.random_select)

    model = Multi_Granularity(layer_num=args.layer_num, device=device)
    model.to(device)


    model.load_state_dict(torch.load(save_file))
    val_dataloader = DataLoader(test_dataset, batch_size=1)
    model.eval()
    model.Train = False

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

