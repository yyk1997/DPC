'''codes used to train backdoored models on poisoned dataset
'''
import argparse
import os, sys
import time
from lightly import loss
from lightly.models.modules import heads
import torchvision.models
from tqdm import tqdm
from utils import default_args, imagenet
import torch
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import ImageFilter
import random


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR.

    Borrowed from https://github.com/facebookresearch/moco/blob/master/moco/loader.py.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))

        return x

class SimCLRTransform:
    def __init__(self, input_size):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size = 32, scale=(0.2, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.RandomApply([
            #     transforms.RandomRotation(degrees=90)  # 旋转角度根据截图可能需要调整
            # ], p=0.8),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomVerticalFlip(p=0.8),  # p=0.0 表示这个变换实际不会发生
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # 根据截图调整亮度、对比度、饱和度和色调参数
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

input_size = 32
simclr_transform = SimCLRTransform(input_size)


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
args.poison_rate = 0.020
args.cover_rate = 0.005
# args.trigger = "badnet_patch4_dup_32.png"
# args.alpha = 0.2

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

# tools.setup_seed(args.seed)

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

    #existing_transforms = list(simclr_transform)
    #existing_transforms.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    data_transform_aug = simclr_transform

    data_transform = data_transform_aug

    # data_transform_aug = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(32, 4),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
    # ])

    # data_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    # ])

elif args.dataset == 'gtsrb':

    data_transform_aug = simclr_transform

    data_transform = data_transform_aug


    # data_transform_aug = transforms.Compose([
    #     transforms.RandomRotation(15),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    # ])
    #
    # data_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    # ])

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
    learning_rate = 0.4
    batch_size = 512

elif args.dataset == 'gtsrb':

    num_classes = 43
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    milestones = torch.tensor([30, 60])
    learning_rate = 0.01
    batch_size = 512

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
    kwargs = {'num_workers': 16, 'pin_memory': True}

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

    print('dataset : %s' % poisoned_set_img_dir)

    poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
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
        batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

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





# 至此训练数据集准备完毕，开始训练阶段

# 首先对模型进行建模
# Train Code
if args.dataset != 'ember':
    # model = arch(num_classes=num_classes)
    resnet = arch(num_classes=num_classes)
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = SimCLR(backbone)

else:
    model = arch()



# name = 'poisoned_train_set/cifar10/trojan_0.100_poison_seed=0'
# model_file = Path("pretrain_model") / name / "pretrain_model_300_epochs.pth"
#
# model.load_state_dict(torch.load(model_file))


# nn.DataParallel的目的是在多个GPU上运行
#model = nn.DataParallel(model)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

# 如果训练数据集不是ember，则采取以下loss设置方法
if args.dataset != 'ember':

    # imagenet与其他数据集一样均使用交叉熵损失函数
    if args.dataset == 'imagenet':
        criterion = loss.NTXentLoss(temperature=0.5)
    else:
        criterion = loss.NTXentLoss(temperature=0.5)
else:
    model_path = os.path.join('poisoned_train_set', 'ember', args.ember_options, 'full_base_aug_seed=%d.pt' % args.seed)
    print(f"Will save to '{model_path}'.")
    if os.path.exists(model_path):
        print(f"Model '{model_path}' already exists!")
    criterion = nn.BCELoss().to(device)

optimizer = torch.optim.SGD(model.parameters(), 0.4, momentum=momentum, weight_decay=weight_decay)
# 训练epoch在milestone时学习率会将会降到原来的10%
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5, verbose=True)
scheduler = CosineAnnealingLR(optimizer, T_max=1000)

if args.poison_type == 'TaCT' or args.poison_type == 'SleeperAgent':
    source_classes = [config.source_class]
else:
    source_classes = None

import time

st = time.time()

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
        x0, x1 = batch[0]
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss = criterion(z0, z1)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()


    avg_loss = total_loss / len(poisoned_set_loader)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print('<Backdoor Training> Train Epoch: {} \tLoss: {:.6f}, lr: {:.6f}, Time: {:.2f}s'.format(epoch, loss.item(),
                                                                                                 optimizer.param_groups[
                                                                                                     0]['lr'],
                                                                                                 elapsed_time))
    scheduler.step()


    if epoch % 100 == 0:
        print("Saving Pretrain model...")
        name = 'poisoned_train_set/gtsrb/adaptive_patch_0.020_cover=0.005_poison_seed=0'
        output_dir = Path('pretrain_model') / name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"pretrain_model_{epoch}_epochs.pth"
        torch.save(model.state_dict(), output_dir / output_name)










#     # Test
#
#     if args.dataset != 'ember':
#         if True:
#             # if epoch % 5 == 0:
#             if args.dataset == 'imagenet':
#                 tools.test_imagenet(model=model, test_loader=test_set_loader,
#                                     test_backdoor_loader=test_set_backdoor_loader)
#                 torch.save(model.module.state_dict(), supervisor.get_model_dir(args))
#             else:
#                 tools.test(model=model, test_loader=test_set_loader, poison_test=True,
#                            poison_transform=poison_transform, num_classes=num_classes, source_classes=source_classes,
#                            all_to_all=all_to_all)
#                 torch.save(model.module.state_dict(), supervisor.get_model_dir(args))
#     else:
#
#         tools.test_ember(model=model, test_loader=test_set_loader,
#                          backdoor_test_loader=backdoor_test_set_loader)
#         torch.save(model.module.state_dict(), model_path)
#     print("")
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