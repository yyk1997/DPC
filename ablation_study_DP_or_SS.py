'''codes used to train backdoored models on poisoned dataset
'''
import argparse
import os, sys
import time
import config
import numpy as np
from lightly import loss
from lightly.models.modules import heads
import torchvision.models
from tqdm import tqdm
from utils import default_args, imagenet
import torch
import math
import random
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


# Create a PyTorch module for the SimCLR model.
class SimCLR(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=512,  # Resnet18 features have 512 dimensions.
            hidden_dim=512,
            output_dim=128,
        )

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str, required=False,
                    default='none',
                    choices=default_args.parser_choices['poison_type'])
parser.add_argument('-poison_rate', type=float, required=False,
                    choices=default_args.parser_choices['poison_rate'],
                    default=default_args.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float, required=False,
                    choices=default_args.parser_choices['cover_rate'],
                    default=default_args.parser_default['cover_rate'])
parser.add_argument('-ember_options', type=str, required=False,
                    choices=['constrained', 'unconstrained', 'none'],
                    default='unconstrained')
parser.add_argument('-alpha', type=float, required=False,
                    default=default_args.parser_default['alpha'])
parser.add_argument('-test_alpha', type=float, required=False, default=None)
parser.add_argument('-trigger', type=str, required=False,
                    default=None)
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-log', default=False, action='store_true')
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)

args = parser.parse_args()

args.dataset = "gtsrb"
args.poison_type = "adaptive_patch"
args.poison_rate = 0.100
args.cover_rate = 0.005

args.debug_info = True
args.no_aug = True
name = 'poisoned_train_set/gtsrb/adaptive_patch_0.100_cover=0.005_poison_seed=0'
name1 = 'pretrain_model_100_epochs.pth'
# args.trigger = "badnet_patch4_dup_32.png"
# args.alpha = 0.2


MAX_GRAD_NORM = 1.0
Noise_Multi = 0.1
DELTA = 1e-5
filter_percent_begin = 0.1
filter_percent = 0.1
params = config.get_params(args)


os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices

import config
from torchvision import datasets, transforms
from torch import nn
import torch
from utils import supervisor, tools

if args.trigger is None:
    if args.dataset != 'imagenet':
        args.trigger = config.trigger_default[args.poison_type]
    elif args.dataset == 'imagenet':
        args.trigger = imagenet.triggers[args.poison_type]

all_to_all = False
if args.poison_type == 'badnet_all_to_all':
    all_to_all = True

tools.setup_seed(args.seed)

if args.log:
    out_path = 'logs'
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, 'base')
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_%s.out' % (
    supervisor.get_dir_core(args, include_poison_seed=config.record_poison_seed), 'no_aug' if args.no_aug else 'aug'))
    fout = open(out_path, 'w')
    ferr = open('/dev/null', 'a')
    sys.stdout = fout
    sys.stderr = ferr

if args.dataset == 'cifar10':

    data_transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
    ])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])

elif args.dataset == 'gtsrb':

    data_transform_aug = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

elif args.dataset == 'imagenet':
    print('[ImageNet]')

elif args.dataset == 'ember':
    print('[Non-image Dataset] Ember')
else:
    raise NotImplementedError('dataset %s not supported' % args.dataset)

if args.dataset == 'cifar10':

    num_classes = 10
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    milestones = torch.tensor([50, 75])
    learning_rate = 0.1
    batch_size = 512

elif args.dataset == 'gtsrb':

    num_classes = 43
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    milestones = torch.tensor([30, 60])
    learning_rate = 0.01
    batch_size = 128

elif args.dataset == 'imagenet':

    num_classes = 1000
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 90
    milestones = torch.tensor([30, 60])
    learning_rate = 0.1
    batch_size = 256

elif args.dataset == 'ember':

    num_classes = 2
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-6
    epochs = 10
    learning_rate = 0.1
    milestones = torch.tensor([])
    batch_size = 512

else:

    print('<Undefined Dataset> Dataset = %s' % args.dataset)
    raise NotImplementedError('<To Be Implemented> Dataset = %s' % args.dataset)

if args.dataset == 'imagenet':
    kwargs = {'num_workers': 32, 'pin_memory': True}
else:
    kwargs = {'num_workers': 4, 'pin_memory': True}

# Set Up Poisoned Set 设置中毒集合

# 如果数据集不是ember，也不是imagenet
if args.dataset != 'ember' and args.dataset != 'imagenet':

    # e.g. poison_set_dir:'poisoned_train_set/cifar10/badnet_0.100_poison_seed=0'
    poison_set_dir = supervisor.get_poison_set_dir(args)

    # e.g. poisoned_set_img_dir: 'poisoned_train_set/cifar10/badnet_0.100_poison_seed=0/data'
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')

    # e.g. poisoned_set_img_dir: 'poisoned_train_set/cifar10/badnet_0.100_poison_seed=0/labels'
    poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')

    # e.g. poisoned_set_img_dir: 'poisoned_train_set/cifar10/badnet_0.100_poison_seed=0/poison_indices'
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')

    poison_indices = torch.load(poison_indices_path)
    print('dataset : %s' % poisoned_set_img_dir)

    poisoned_set = tools.IMG_Dataset_1(data_dir=poisoned_set_img_dir,
                                     label_path=poisoned_set_label_path,
                                     transforms=data_transform if args.no_aug else data_transform_aug)

    # 中毒数据集的loader
    poisoned_set_loader = torch.utils.data.DataLoader(
        poisoned_set,
        batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

elif args.dataset == 'imagenet':

    poison_set_dir = supervisor.get_poison_set_dir(args)
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    print('dataset : %s' % poison_set_dir)

    poison_indices = torch.load(poison_indices_path)

    root_dir = '/path_to_imagenet/'
    train_set_dir = os.path.join(root_dir, 'train')
    test_set_dir = os.path.join(root_dir, 'val')

    from utils import imagenet

    poisoned_set = imagenet.imagenet_dataset(directory=train_set_dir, poison_directory=poisoned_set_img_dir,
                                             poison_indices=poison_indices, target_class=imagenet.target_class,
                                             num_classes=1000)

    poisoned_set_loader = torch.utils.data.DataLoader(
        poisoned_set,
        batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs, drop_last = False)

else:
    poison_set_dir = os.path.join('poisoned_train_set', 'ember', args.ember_options)
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')

    # stats_path = os.path.join('data', 'ember', 'stats')
    poisoned_set = tools.EMBER_Dataset(x_path=os.path.join(poison_set_dir, 'watermarked_X.npy'),
                                       y_path=os.path.join(poison_set_dir, 'watermarked_y.npy'))
    print('dataset : %s' % poison_set_dir)

    poisoned_set_loader = torch.utils.data.DataLoader(
        poisoned_set,
        batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)



#同等对测试数据集合进行设置，目标是得到test_set_loader
if args.dataset != 'ember' and args.dataset != 'imagenet':

    # Set Up Test Set for Debug & Evaluation
    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_set_img_dir = os.path.join(test_set_dir, 'data')
    test_set_label_path = os.path.join(test_set_dir, 'labels')
    test_set = tools.IMG_Dataset_1(data_dir=test_set_img_dir,
                                 label_path=test_set_label_path, transforms=data_transform)
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    # Poison Transform for Testing
    poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                       target_class=config.target_class[args.dataset],
                                                       trigger_transform=data_transform,
                                                       is_normalized_input=True,
                                                       alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                       trigger_name=args.trigger, args=args)


elif args.dataset == 'imagenet':

    poison_transform = imagenet.get_poison_transform_for_imagenet(args.poison_type)

    test_set = imagenet.imagenet_dataset(directory=test_set_dir, shift=False, aug=False,
                                         label_file=imagenet.test_set_labels, num_classes=1000)
    test_set_backdoor = imagenet.imagenet_dataset(directory=test_set_dir, shift=False, aug=False,
                                                  label_file=imagenet.test_set_labels, num_classes=1000,
                                                  poison_transform=poison_transform)

    test_split_meta_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_indices = torch.load(os.path.join(test_split_meta_dir, 'test_indices'))

    test_set = torch.utils.data.Subset(test_set, test_indices)
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    test_set_backdoor = torch.utils.data.Subset(test_set_backdoor, test_indices)
    test_set_backdoor_loader = torch.utils.data.DataLoader(
        test_set_backdoor,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

else:
    normalizer = poisoned_set.normal

    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')

    test_set = tools.EMBER_Dataset(x_path=os.path.join(test_set_dir, 'X.npy'),
                                   y_path=os.path.join(test_set_dir, 'Y.npy'),
                                   normalizer=normalizer)

    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    backdoor_test_set_dir = os.path.join('poisoned_train_set', 'ember', args.ember_options)
    backdoor_test_set = tools.EMBER_Dataset(x_path=os.path.join(poison_set_dir, 'watermarked_X_test.npy'),
                                            y_path=None, normalizer=normalizer)
    backdoor_test_set_loader = torch.utils.data.DataLoader(
        backdoor_test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)




import confusion_training


def iterative_poison_distillation(inspection_set, clean_set, params, args, distillation_ratio, debug_packet=None, start_iter=0 ,momentums = params['momentums'],lambs = params['lambs'],lrs = params['lrs']):

    if args.debug_info and (debug_packet is None):
        raise Exception('debug_packet is needed to compute debug info')

    # 'kwargs' : {'num_workers': 2, 'pin_memory': True},
    kwargs = params['kwargs']
    # 获取中毒数据集合
    inspection_set_dir = params['inspection_set_dir']
    # cifar10是10类，gtsrb是43类
    num_classes = params['num_classes']
    # 'pretrain_epochs' : 100,在该函数中没用上
    pretrain_epochs = 10
    # pretrain_epochs = params['pretrain_epochs']
    # 'weight_decay' : 1e-4,
    weight_decay = params['weight_decay']
    # 依据'arch' :  arch[args.dataset],从而获取模型架构
    arch = params['arch']

    batch_factor = params['batch_factors']

    # 打印模型
    print('arch = ', arch)

    # 设立干净数据的dataloader
    clean_set_loader = torch.utils.data.DataLoader(
        # batch_size = 32
        clean_set, batch_size=params['batch_size'],
        shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

    # clean_set_loader_aug = torch.utils.data.DataLoader(
    #    clean_set_aug, batch_size=params['batch_size'],
    #    shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

    # 开始迭代执行训练
    print('>>> Iterative Data Distillation with Confusion Training')

    # 蒸馏样本索引和中位数样本索引均设置为None
    distilled_samples_indices, median_sample_indices = None, None
    # confusion迭代的次数设置，一共迭代调整5次，所以迭代6次
    num_confusion_iter = len(distillation_ratio) + 1
    #num_confusion_iter = len(distillation_ratio)

    # reduction=none表示需要计算每个样本的损失值
    criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')

    criterion = nn.CrossEntropyLoss()

    # 如果开始迭代的循环次数不是0，则首先开始蒸馏出一部分样本索引
    if start_iter != 0:
        distilled_samples_indices, _ = confusion_training.distill(args, params, inspection_set,
                                                                  start_iter - 1, criterion_no_reduction)
        distilled_set = torch.utils.data.Subset(inspection_set, distilled_samples_indices)

    else:
        # 蒸馏数据集初始化为整个集合
        distilled_set = inspection_set

    # start_iter=0开始，num_confusion_iter=6
    for confusion_iter in range(start_iter, num_confusion_iter):

        size_of_distilled_set = len(distilled_set)
        print('<Round-%d> Size_of_distillation_set = ' % confusion_iter, size_of_distilled_set)

        # different weights for each class based on their frequencies in the distilled set
        # 由于每一类中数据的数量是不同的，因此要对每一类数据赋予不同的权重，下面方法用于计算权重
        nums_of_each_class = np.zeros(num_classes)
        for i in range(size_of_distilled_set):
            _, gt = distilled_set[i]
            gt = gt.item()
            nums_of_each_class[gt] += 1
        # 计算得到每一类中数据的条目数
        print(nums_of_each_class)

        # 执行频率归一化处理
        freq_of_each_class = nums_of_each_class / size_of_distilled_set
        # 执行频率的平方根变换？，且由于加入了0.001，使得变换后的频率不会出错，估计开根号是一种数据平滑性的操作，拉近各类数据频率的值，使得它们之间的差异性不会特别突出
        freq_of_each_class = np.sqrt(freq_of_each_class + 0.001)

        # 按照不同的迭代次数对pretrain_epochs, pretrain_lr, distillation_iters 进行设置
        if confusion_iter < 2:  # lr=0.01 for round 0,1,2
            pretrain_epochs = 100
            pretrain_lr = 0.01
            distillation_iters = 375
            # distillation_iters = 6000 32 * 6000 = 512 * 375
        elif confusion_iter < 3:  # lr=0.01 for round 0,1,2
            pretrain_epochs = 40
            pretrain_lr = 0.01
            distillation_iters = 375
            # distillation_iters = 6000
        elif confusion_iter < 4:
            pretrain_epochs = 40
            pretrain_lr = 0.01
            distillation_iters = 375
            # distillation_iters = 6000
        elif confusion_iter < 5:
            pretrain_epochs = 40
            pretrain_lr = 0.01
            distillation_iters = 375
            # distillation_iters = 2000 32 * 2000 = 512 * 128
        else:
            pretrain_epochs = 40
            pretrain_lr = 0.01
            distillation_iters = 375
            # distillation_iters = 2000

        # 根据蒸馏次数还设置了不同的学习率，蒸馏的前几轮学习率低，后续逐渐升高
        lr = lrs[confusion_iter]

        # 当执行到最后一轮蒸馏时，所有类的频率全部被设置为1
        if confusion_iter == num_confusion_iter - 1:
            freq_of_each_class[:] = 1

        # 当没有执行到最后一轮时，用distill_set设置distill_set_loader,否则将distill_set和clean_set联合起来一起设置
        if confusion_iter != num_confusion_iter - 1:
            distilled_set_loader = torch.utils.data.DataLoader(
                distilled_set,
                batch_size=params['batch_size'], shuffle=True,
                worker_init_fn=tools.worker_init, **kwargs)
        else:
            distilled_set_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset([distilled_set, clean_set]),
                batch_size=params['batch_size'], shuffle=True,
                worker_init_fn=tools.worker_init, **kwargs)

        # 输出每一类数据的频率
        print('freq: ', freq_of_each_class)

        # pretrain base model，先拟合出一个中毒的模型, return model
        confusion_training.pretrain(args, debug_packet, arch, num_classes, weight_decay, pretrain_epochs,
                                    distilled_set_loader, criterion, inspection_set_dir, confusion_iter,
                                    pretrain_lr)

        distilled_set_loader = torch.utils.data.DataLoader(
            distilled_set,
            batch_size=params['batch_size'], shuffle=True,
            worker_init_fn=tools.worker_init, **kwargs)

        # confusion_training
        model = confusion_training.confusion_train(args, params, inspection_set, debug_packet,
                                                   distilled_set_loader, clean_set_loader, confusion_iter, arch,
                                                   num_classes, inspection_set_dir, weight_decay,
                                                   criterion_no_reduction,
                                                   momentums[confusion_iter], lambs[confusion_iter],
                                                   freq_of_each_class, lr, batch_factor[confusion_iter],
                                                   distillation_iters)

        # distill the inspected set according to the loss values
        distilled_samples_indices, median_sample_indices = confusion_training.distill(args, params,
                                                                                      inspection_set,
                                                                                      confusion_iter,
                                                                                      criterion_no_reduction, distillation_ratio)



        distilled_set = torch.utils.data.Subset(inspection_set, distilled_samples_indices)


        # distilled_set_aug = torch.utils.data.Subset(inspection_set_aug, distilled_samples_indices)



    print(len(distilled_samples_indices))

    return distilled_samples_indices, median_sample_indices, model






# 至此训练数据集准备完毕，开始训练阶段
 # 建立10个空列表
# lists 用来存储每一类的数据
lists = [[] for _ in range(params['num_classes'])]

# 用来存放每一类数据在全局层面上的索引
lists_overall_indices = [[] for _ in range(params['num_classes'])]

every_kind_frequency = []

for i, (image, label) in enumerate(zip(poisoned_set.images, poisoned_set.gt)):
    # lists中对应的类列表添加图像
    lists[label].append(image.clone().detach())
    lists_overall_indices[label].append(i)
    # if i in set(poison_indices):
    #     lists_poison_indices.append(len(lists[label]) - 1)
print("10类数据分配完毕")

for l in lists:
    every_kind_frequency.append(len(l) / len(poisoned_set.gt))


device = "cuda:0" if torch.cuda.is_available() else "cpu"
# 首先对模型进行建模
# Train Code
if args.dataset != 'ember':
    # model = arch(num_classes=num_classes)
    resnet = arch(num_classes=num_classes)
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = SimCLR(backbone)
    model_file = Path("pretrain_model") / name / name1
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    model.projection_head = nn.Linear(512, 43).to(device)
    # 设置 backbone 中的所有参数不进行梯度更新
    for param in model.backbone.parameters():
        param.requires_grad = False

    # 设置 projection_head 中的参数进行梯度更新
    for param in model.projection_head.parameters():
        param.requires_grad = True



else:
    model = arch()


milestones = milestones.tolist()

# nn.DataParallel的目的是在多个GPU上运行
##model = nn.DataParallel(model)

model.to(device)


# 加载训练好的中毒模型
poison_model = params['arch'](num_classes = num_classes)
inspection_set_dir = params['inspection_set_dir']
poison_model.load_state_dict(
            torch.load(os.path.join(inspection_set_dir, 'full_base_aug_seed=%d_15epochs.pt' % (args.seed)))
        )

poison_model = poison_model.to(device)
poison_model.eval()



# 如果训练数据集不是ember，则采取以下loss设置方法
if args.dataset != 'ember':
    #print(f"Will save to '{supervisor.get_model_dir(args)}'.")
    #print(f"Will save to 'poisoned_train_set/cifar10/badnet_0.100_poison_seed=0/pretrain.pt'.")
    # if os.path.exists(supervisor.get_model_dir(args)):
    #     print(f"Model '{supervisor.get_model_dir(args)}' already exists!")

    # imagenet与其他数据集一样均使用交叉熵损失函数
    if args.dataset == 'imagenet':
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)
else:
    model_path = os.path.join('poisoned_train_set', 'ember', args.ember_options, 'full_base_aug_seed=%d.pt' % args.seed)
    print(f"Will save to '{model_path}'.")
    if os.path.exists(model_path):
        print(f"Model '{model_path}' already exists!")
    criterion = nn.BCELoss().to(device)

# optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
# # 训练epoch在milestone时学习率会将会降到原来的10%
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5, verbose=True)

if args.poison_type == 'TaCT' or args.poison_type == 'SleeperAgent':
    source_classes = config.source_class
else:
    source_classes = None



import time

st = time.time()
list_intersection = []


# # DP化实现
# privacy_engine = PrivacyEngine(
# )
# model.train()
# model = ModuleValidator.fix(model)
# optimizer = torch.optim.SGD(model.projection_head.parameters(), 0.01, momentum=momentum, weight_decay=weight_decay)
# model, optimizer, poisoned_set_loader = privacy_engine.make_private(
#                 module=model,
#                 optimizer=optimizer,
#                 data_loader=poisoned_set_loader,
#                 max_grad_norm=MAX_GRAD_NORM,
#                 noise_multiplier=Noise_Multi
#             )

optimizer = torch.optim.SGD(model.projection_head.parameters(), 0.1, momentum=momentum, weight_decay=weight_decay)


all_item_list = list(range(len(poisoned_set.gt)))
suspicious_indices_set_total = set()
distilled_samples_indices_set = set()
inspection_indices = list(range(len(poisoned_set.gt)))
inspection_set = poisoned_set
poison_indices = torch.load(os.path.join(params['inspection_set_dir'], 'poison_indices'))

index = 0
# scaler = GradScaler()
for epoch in range(1, epochs + 1):  # train backdoored base model
    start_time = time.perf_counter()
    total_loss = 0
    # Train
    model.train()
    preds = []
    labels = []
    for batch in tqdm(poisoned_set_loader):

        optimizer.zero_grad()

        # 获取当前批次的图像数据和标签
        batch_img = batch[0]
        batch_label = batch[1]
        batch_img = batch_img.to(device)
        batch_label = batch_label.to(device)

        # 模型前向传播，得到预测结果
        predict_digits = model(batch_img)
        # predict_digits = model.fc(predict_digits)

        # 计算损失
        loss = criterion(predict_digits, batch_label)
        total_loss += loss
        # 反向传播，计算梯度
        loss.backward()
        # 模型更新参数
        optimizer.step()

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

    avg_loss = total_loss / len(poisoned_set_loader)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print('<Backdoor Training> Train Epoch: {} \tLoss: {:.6f}, lr: {:.6f}, Time: {:.2f}s'.format(epoch, avg_loss.item(),
                                                                                                 optimizer.param_groups[
                                                                                                     0]['lr'],
                                                                                                 elapsed_time))
    #scheduler.step(avg_loss)
    #scheduler.step()

    # 以上代码为模型训练

    # 下面代码为数据筛选
    # 首先进行在robust training部分利用loss进行筛选

    if epoch % 1 == 0:


        model.eval()
        first_one_second_percent_total = set()
        #for i in range(params['num_classes']):
        target_label_loss = []
        for target_label_sample in lists[2]:
            sample_tensor = target_label_sample.unsqueeze(0)
            sample_tensor = sample_tensor.to(device)
            with torch.no_grad():
                predict_digits = model(sample_tensor)
                target = torch.tensor([2], device=predict_digits.device)
                loss = criterion(predict_digits, target)
                target_label_loss.append(loss)

        #sorted_indices = sorted(range(len(target_label_loss)), key=lambda x: target_label_loss[x])

        sorted_overall_indices = sorted(zip(target_label_loss, lists_overall_indices[2]))
        sorted_overall_indices = [item[1] for item in sorted_overall_indices]

        first_filter_count = int(len(sorted_overall_indices) * filter_percent_begin)

        first_filter_indices_set = set(sorted_overall_indices[:first_filter_count])

        intersection = first_filter_indices_set & set(poison_indices)
        print(
            f"第{2}类数据中 loss 值前{filter_percent_begin}中，总样本数量为{first_filter_count}个，其中有中毒样本数量：{len(intersection)}")

        # 其次在中毒模型poison_model上依照loss值进行筛选
        first_filter_indices_list = list(first_filter_indices_set)


        target_clean_sample_loss = []
        for item in first_filter_indices_list:
            sample = lists[2][lists_overall_indices[2].index(item)]
            sample_tensor = sample.unsqueeze(0)
            sample_tensor = sample_tensor.to(device)
            with torch.no_grad():
                predict_digits = poison_model(sample_tensor)
                target = torch.tensor([2], device=predict_digits.device)
                loss = criterion(predict_digits, target)
                target_clean_sample_loss.append(loss)

        sorted_overall_clean_indices = sorted(zip(target_clean_sample_loss, first_filter_indices_list))
        sorted_overall_clean_indices = [item[1] for item in sorted_overall_clean_indices]

        ten_percent_count = int(len(sorted_overall_clean_indices) * filter_percent)

        for j in range(10):
            one_second_percent = set(
                sorted_overall_clean_indices[j * ten_percent_count: (j + 1) * ten_percent_count])
            intersection = one_second_percent & set(poison_indices)
            print(
                f"在上述{filter_percent_begin}%的基础上 第{0}类数据中 以poison_model计算loss值从低到高的第{j}个10%item{ten_percent_count}个item中，有中毒样本数量：{len(intersection)}")

        first_one_second_percent = set(sorted_overall_clean_indices[9 * ten_percent_count:10 * ten_percent_count])

            # if index == 0:
            #     first_one_second_percent = set(sorted_overall_clean_indices[9 * ten_percent_count:10 * ten_percent_count])
            #
            # elif index == 1:
            #     first_one_second_percent = set(sorted_overall_clean_indices[-100:])
            #     # first_one_second_percent = set(sorted_overall_clean_indices[
            #     #                                math.ceil(4.5 * ten_percent_count):math.ceil(
            #     #                                    5.5 * ten_percent_count)])
            # if index < 2:
            #     if index < 1:
            #         first_one_second_percent = set(
            #         sorted_overall_clean_indices[9 * ten_percent_count: 10 * ten_percent_count])
            #     else:
            #         first_one_second_percent = set(sorted_overall_clean_indices[
            #                                    math.ceil(4.5 * ten_percent_count):math.ceil(
            #                                        5.5 * ten_percent_count)])
            # else:
            #     # if len(sorted_overall_clean_indices) >= 500:
            #     #     first_one_second_percent = set(random.sample(sorted_overall_clean_indices, 500))
            #     # else:
            #     #     first_one_second_percent = set(sorted_overall_clean_indices)
            #
            #     num = int(math.ceil(5000 * every_kind_frequency[i]))
            #     print("num: ", num)
            #     if num <= len(sorted_overall_clean_indices):
            #         first_one_second_percent = set(random.sample(sorted_overall_clean_indices, num))
            #     else:
            #         first_one_second_percent = set(sorted_overall_clean_indices)




        first_one_second_percent_total = first_one_second_percent_total | first_one_second_percent

        print("len(first_one_second_percent_total: {}".format(len(first_one_second_percent_total)))

        print("first_one_second_percent_total(clean_set)中有多少中毒样本：")
        print(len(first_one_second_percent_total & set(poison_indices)))

        # first_one_second_percent_total_indices = list(first_one_second_percent_total)
        # first_one_second_percent_total_indices.sort()
        #
        # clean_data_set = torch.utils.data.Subset(poisoned_set, first_one_second_percent_total_indices)
        #
        #
        # debug_packet = None
        # if args.debug_info:
        #     debug_packet = config.get_packet_for_debug(params['inspection_set_dir'], params['data_transform'],
        #                                                params['batch_size'], args)
        #
        # if index == 2:
        #     # clean_indices = random.sample(inspection_indices, 5000)
        #     #clean_indices = inspection_indices
        #
        #     print("len(clean_indices):", len(first_one_second_percent_total_indices))
        #     print("候选干净样本集中有多少中毒样本：")
        #     print(len(set(first_one_second_percent_total_indices) & set(poison_indices)))
        #     print("候选干净样本中中毒样本个数占中毒样本的比例是多少：")
        #     print(len(set(first_one_second_percent_total_indices) & set(poison_indices)) / len(poison_indices))
        #
        #     #clean_data_set = torch.utils.data.Subset(poisoned_set, clean_indices)
        #
        #
        #     distilled_samples_indices, median_sample_indices, confuse_model = iterative_poison_distillation(
        #         poisoned_set,
        #         clean_data_set, params, args, [1/2,1/5,1/25,1/50, 1/100],
        #         debug_packet,
        #         start_iter=0,)
        #
        #     # 识别中毒样本
        #     print('to identify poison samples')
        #     # detect backdoor poison samples with the confused model
        #     suspicious_indices = confusion_training.identify_poison_samples_simplified(poisoned_set,
        #                                                                                median_sample_indices,
        #                                                                                confuse_model,
        #                                                                                num_classes=params[
        #                                                                                    'num_classes'])
        #
        #
        #
        # else:
        #
        #     distilled_samples_indices, median_sample_indices, confuse_model = iterative_poison_distillation(poisoned_set,
        #                                                                                         clean_data_set, params, args,[],
        #                                                                                         debug_packet,
        #                                                                                        start_iter=0,lambs = [20],lrs = [0.001],
        #                                                                                                     momentums = [0.7])
        #
        #     print(len(distilled_samples_indices))
        #     suspicious_indices = distilled_samples_indices
        #
        #
        #
        # suspicious_indices_set = set(suspicious_indices)
        # suspicious_indices_set_list = list(suspicious_indices_set)
        # if args.debug_info:
        #     suspicious_indices_set_list.sort()
        #
        #     num_samples = len(poisoned_set)
        #     num_poison = len(poison_indices)
        #     num_collected = len(suspicious_indices_set_list)
        #     pt = 0
        #     recall = 0
        #     for idx in suspicious_indices_set_list:
        #         if pt >= num_poison:
        #             break
        #         while (idx > poison_indices[pt] and pt + 1 < num_poison): pt += 1
        #         if pt < num_poison and poison_indices[pt] == idx:
        #             recall += 1
        #     fpr = num_collected - recall
        #
        #     print('recall = %d/%d = %f, fpr = %d/%d = %f' % (
        #         recall, num_poison, recall / num_poison if num_poison != 0 else 0,
        #         fpr, num_samples - num_poison,
        #         fpr / (num_samples - num_poison) if (num_samples - num_poison) != 0 else 0))
        #
        #
        #
        #
        # inspection_indices = [x for x in all_item_list if x not in suspicious_indices_set_list]
        #
        # inspection_indices.sort()
        #
        # if index == 2:
        #     break
        #
        #
        # # 建立10个空列表
        # # lists 用来存储每一类的数据
        # lists = [[] for _ in range(params['num_classes'])]
        #
        # # 用来存放每一类数据在全局层面上的索引
        # lists_overall_indices = [[] for _ in range(params['num_classes'])]
        #
        # for i, (image, label) in enumerate(zip(poisoned_set.images, poisoned_set.gt)):
        #     if i not in suspicious_indices_set:
        #         # lists中对应的类列表添加图像
        #         lists[label].append(image.clone().detach())
        #         lists_overall_indices[label].append(i)
        #
        # print("10类数据分配完毕")
        #
        # print("筛选到中毒样本")
        # print(len(suspicious_indices_set & set(poison_indices)))
        #
        # # print("蒸馏之后留下的数据个数：")
        # # print(len(inspection_indices))
        #
        # inspection_set = torch.utils.data.Subset(poisoned_set, inspection_indices)
        #
        # poisoned_set_loader = torch.utils.data.DataLoader(
        #     inspection_set,
        #     batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)
        #
        # MAX_GRAD_NORM = MAX_GRAD_NORM * 2
        # print("MAX_GRAD_NORM: ", MAX_GRAD_NORM)
        # Noise_Multi = Noise_Multi / 2
        # print("Noise_Multi:", Noise_Multi)
        # privacy_engine = PrivacyEngine(
        # )
        # model.train()
        # model, optimizer, poisoned_set_loader = privacy_engine.make_private(
        #     module=model,
        #     optimizer=optimizer,
        #     data_loader=poisoned_set_loader,
        #     max_grad_norm=MAX_GRAD_NORM,
        #     noise_multiplier=Noise_Multi
        # )
        #
        # if index < 1:
        #     filter_percent_begin = filter_percent_begin * 5
        #
        # index += 1

    # Test

    if args.dataset != 'ember':
        if True:
            #if epoch == 1 or epoch % 5 == 0:
                if args.dataset == 'imagenet':
                    tools.test_imagenet(model=model, test_loader=test_set_loader,
                                    test_backdoor_loader=test_set_backdoor_loader)
                #torch.save(model.module.state_dict(), supervisor.get_model_dir(args))
                else:
                    tools.test(model=model, test_loader=test_set_loader, poison_test=True,
                               poison_transform=poison_transform, num_classes=num_classes, source_classes=source_classes,
                               all_to_all=all_to_all)
                    #torch.save(model.module.state_dict(), supervisor.get_model_dir(args))
    else:

        tools.test_ember(model=model, test_loader=test_set_loader,
                         backdoor_test_loader=backdoor_test_set_loader)
        #torch.save(model.module.state_dict(), model_path)
    print("")


inspection_indices.sort()
save_path = os.path.join(poison_set_dir, 'DP_ct_cleansed_set_indices_seed=%d' % args.seed)
torch.save(inspection_indices, save_path)
print('[Save] %s' % save_path)











#
# if args.dataset != 'ember':
#     torch.save(model.module.state_dict(), supervisor.get_model_dir(args))
# else:
#     torch.save(model.module.state_dict(), model_path)
#
# if args.poison_type == 'none':
#     if args.no_aug:
#         torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_no_aug.pt')
#         torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_no_aug_seed={args.seed}.pt')
#     else:
#         torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_aug.pt')
#         torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_aug_seed={args.seed}.pt')