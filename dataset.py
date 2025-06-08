import json
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class Countgwhd(Dataset):
    def __init__(self, img_path, ann_path, density_path ,clip_json, resize_shape, number=3,random_select=False,
                 imageHeight = 1024,imageWidth =1024, boxHeight =100, boxWidth=100, negative=False, negative_json = " ",
                 negative_num = 10, crop_size=50, image_name_out = False):
        self.img_path = img_path
        self.ann_path = pd.read_csv(ann_path)
        self.shape = resize_shape
        self.transform = transforms.Resize([self.shape, self.shape])
        self.number = number
        self.random_select = random_select
        self.density_path = density_path
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
        self.boxHeight = int(boxHeight/2)
        self.boxWidth = int(boxWidth/2)
        self.negative = negative
        self.negative_num = negative_num
        self.crop_size = crop_size
        self.image_name_out = image_name_out
        print("negative_num:  {}".format(self.negative_num))
        print("麦穗框大小：{}".format(self.crop_size * 2))
        with open(clip_json, "r") as file:
            self.clip_json = json.load(file)
        if self.negative:
            with open(negative_json,"r") as file1:
                self.negative_json = json.load(file1)

    def __getitem__(self, idex):
        # 拼接图片
        img_path = os.path.join(self.img_path + self.ann_path.iloc[idex, 0])
        # tensor类型
        image = Image.open(img_path)
        image_name = self.ann_path.iloc[idex, 0]
        image = image.convert("RGB")
        TOtensor = transforms.ToTensor()
        image = TOtensor(image)

        #裁剪示例图像
        rect = self.select_sample(self.ann_path.iloc[idex, 0])

        crop_image = image[:,rect[1]:rect[3],rect[0]:rect[2]]

        Tosize = transforms.Resize([224, 224])
        crop_image = Tosize(crop_image)
        crop_image = crop_image.unsqueeze(0)

        if self.negative:
            cropped_negatives = []
            point_num = 0
            for point in self.negative_json[self.ann_path.iloc[idex, 0]]["negative_points"]:
                left = point[0] - self.crop_size
                upper = point[1] - self.crop_size
                right = point[0] + self.crop_size
                lower = point[1] + self.crop_size
                cropped_negative = image[:, upper:lower, left:right]
                cropped_negative = Tosize(cropped_negative)
                cropped_negatives.append(cropped_negative)
                point_num +=1
                if point_num > self.negative_num:
                    break
            cropped = torch.stack(cropped_negatives)
            crop_image = torch.cat([crop_image, cropped],dim=0)


        # label = self.ann_path.iloc[idex, 1]
        label = np.load(os.path.join(self.density_path,self.ann_path.iloc[idex, 0].replace('.png','.npy')))
        label = torch.tensor(label).unsqueeze(0)
        image = self.transform(image)

        if self.image_name_out:
            return image, label, crop_image, image_name
        else:
            return image, label, crop_image

    def __len__(self):
            return len(self.ann_path)


    def select_sample(self, name):
        if len(self.clip_json[name]["points"]) == 0:
            x = random.randint(0,1024)
            y = random.randint(0,1024)
            return self.merge_point([[x,y]])
        if len(self.clip_json[name]["points"]) < self.number:
            # 如果小于给定框数量，合并所有框
            return self.merge_point(self.clip_json[name]["points"])
        else:
            random_index = 0
            if self.random_select:
                num_items = len(self.clip_json[name]["points"])
                random_index = random.randrange(num_items)
            # 挑选距离最近框
            near_id = self.points_distance(self.clip_json[name]["points"][random_index], self.clip_json[name]["points"], self.number)
            select_point = [self.clip_json[name]["points"][i] for i in near_id]
            select_point.append(self.clip_json[name]["points"][random_index])
            return self.merge_point(select_point)

    def merge_point(self,points):
        # 把所给点进行合并
        x_min = self.imageWidth
        y_min = self.imageHeight
        x_max = 0
        y_max = 0
        for point in points:
            if (point[0] - self.boxWidth) > 0:
                x_min = min((point[0] - self.boxWidth), x_min)
            if (point[1] - self.boxHeight) > 0:
                y_min = min((point[1] - self.boxHeight), y_min)
            if (point[0] + self.boxWidth) < self.imageWidth:
                x_max = max((point[0] + self.boxWidth), x_max)
            if (point[1] + self.boxHeight) > y_max and (point[1] + self.boxHeight) < self.imageHeight:
                y_max = max((point[1] + self.boxHeight), y_max)

            if x_max == 0:
                x_max = 1024
            if y_max == 0:
                y_max = 1024
            if x_min == 1024:
                x_min = 0
            if y_min == 1024:
                y_min = 0

        return [x_min, y_min, x_max, y_max]

    def points_distance(self,ori_point, points, number):
        # 挑选距离最近的点，并返回索引
        min_distance = [[10000 for _ in range(number)], [-1 for _ in range(number)]]

        for point in points:
            temp = int(((ori_point[0] - point[0]) ** 2 + (ori_point[1] - point[1]) ** 2) ** 0.5)
            if temp < max(min_distance[0]):
                id = min_distance[0].index(max(min_distance[0]))
                min_distance[1][id] = points.index(point)
                min_distance[0][id] = temp
        return min_distance[1]



