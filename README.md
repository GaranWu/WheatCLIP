# WheatCLIP

**WheatCLIP** is a wheat ear counting framework that uses large models and contrastive learning for feature enhancement.

paper: WheatCLIP: Object-Aware Wheat Ear Counting with Contrast Learning and Universal Knowledge Model
## The Overview of WheatCLIP
![](methodoverview.jpg)


## About Data
We use the global wheat Head Detection 2021 ([dataset](http://www.global-wheat.com/gwhd.html)) for training.

## Code Structure
`train.py` To train the model. 

`test.py` To test the model. 

## Training
```shell
python train.py
```
# Testing
```shell
python test.py 
```
# Weight
During training, the weights of the first ten layers of VGG16 need to be loaded.
