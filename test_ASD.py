'''codes used to train backdoored models on poisoned dataset
'''
import argparse
import itertools
import os, sys
import random
import time
from tqdm import tqdm
from pathlib import Path
from utils import default_args, imagenet
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from log import AverageMeter, tabulate_step_meter, tabulate_epoch_meter
import numpy as np
from copy import deepcopy


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
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
args.dataset = "gtsrb"
args.poison_type = "badnet"
args.poison_rate = 0.005
#args.cover_rate = 0.005
# args.trigger = "badnet_patch4_dup_32.png"
# args.alpha = 0.2
# name = 'poisoned_train_set/cifar10/badnet_0.005_poison_seed=0'
# name1 = 'pretrain_model_100_epochs.pth'
num_classes = 43




from torch.utils.data import Dataset
class TwiceAugmentedDataset(Dataset):
    def __init__(self, dataset):
        """
        dataset: 被包装的原始数据集，应当已经包含了带有随机性的预处理操作。
        """
        self.dataset = dataset

    def __getitem__(self, index):
        # 访问原始数据集获取数据和标签
        # 因为数据集的预处理具有随机性，所以即便是同一个index，
        # 两次调用__getitem__也会得到不同的结果
        data1, label = self.dataset[index]
        data2, _ = self.dataset[index]  # 忽略第二次的标签

        return data1, data2

    def __len__(self):
        return len(self.dataset)




from lightly.models.modules import heads
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

class MixMatchLoss(nn.Module):
    """SemiLoss in MixMatch.

    Modified from https://github.com/YU1ut/MixMatch-pytorch/blob/master/train.py.
    """

    def __init__(self, rampup_length, lambda_u=75):
        super(MixMatchLoss, self).__init__()
        self.rampup_length = rampup_length
        self.lambda_u = lambda_u
        self.current_lambda_u = lambda_u

    def linear_rampup(self, epoch):
        if self.rampup_length == 0:
            return 1.0
        else:
            current = np.clip(epoch / self.rampup_length, 0.0, 1.0)
            self.current_lambda_u = float(current) * self.lambda_u

    def forward(self, xoutput, xtarget, uoutput, utarget, epoch):
        self.linear_rampup(epoch)
        uprob = torch.softmax(uoutput, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(xoutput, dim=1) * xtarget, dim=1))
        Lu = torch.mean((uprob - utarget) ** 2)

        return Lx, Lu, self.current_lambda_u



class RCELoss(nn.Module):
    """Reverse Cross Entropy Loss."""

    def __init__(self, num_classes=10, reduction="mean"):
        super(RCELoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, x, target):
        prob = F.softmax(x, dim=-1)
        prob = torch.clamp(prob, min=1e-7, max=1.0)
        one_hot = F.one_hot(target, self.num_classes).float()
        one_hot = torch.clamp(one_hot, min=1e-4, max=1.0)
        loss = -1 * torch.sum(prob * torch.log(one_hot), dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()

        return loss

class SCELoss(nn.Module):
    """Symmetric Cross Entropy."""

    def __init__(self, alpha=0.1, beta=1, num_classes=10, reduction="mean"):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, x, target):
        ce = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        rce = RCELoss(num_classes=self.num_classes, reduction=self.reduction)
        ce_loss = ce(x, target)
        rce_loss = rce(x, target)
        loss = self.alpha * ce_loss + self.beta * rce_loss

        return loss


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
        transforms.RandomCrop(32, 4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
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
    batch_size = 128

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

# Set Up Poisoned Set

if args.dataset != 'ember' and args.dataset != 'imagenet':
    poison_set_dir = supervisor.get_poison_set_dir(args)
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')

    print('dataset : %s' % poisoned_set_img_dir)

    poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                     label_path=poisoned_set_label_path,
                                     transforms=data_transform if args.no_aug else data_transform_aug)

    poisoned_set_loader = torch.utils.data.DataLoader(
        poisoned_set,
        batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

    poisoned_set_static = tools.IMG_Dataset_1(data_dir=poisoned_set_img_dir,
                                     label_path=poisoned_set_label_path,
                                     transforms=data_transform if args.no_aug else data_transform_aug)

    poisoned_set_static_loader = torch.utils.data.DataLoader(
        poisoned_set,
        batch_size = batch_size, shuffle = False, worker_init_fn=tools.worker_init, **kwargs
    )

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

if args.dataset != 'ember' and args.dataset != 'imagenet':

    # Set Up Test Set for Debug & Evaluation
    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_set_img_dir = os.path.join(test_set_dir, 'data')
    test_set_label_path = os.path.join(test_set_dir, 'labels')
    test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
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

# # Train Code
# if args.dataset != 'ember':
#     model = arch(num_classes=num_classes)
# else:
#     model = arch()
#
# milestones = milestones.tolist()
# model = nn.DataParallel(model)
# model = model.cuda()

if args.dataset != 'ember':
    print(f"Will save to '{supervisor.get_model_dir(args)}'.")
    if os.path.exists(supervisor.get_model_dir(args)):
        print(f"Model '{supervisor.get_model_dir(args)}' already exists!")

    if args.dataset == 'imagenet':
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
else:
    model_path = os.path.join('poisoned_train_set', 'ember', args.ember_options, 'full_base_aug_seed=%d.pt' % args.seed)
    print(f"Will save to '{model_path}'.")
    if os.path.exists(model_path):
        print(f"Model '{model_path}' already exists!")
    criterion = nn.BCELoss().cuda()

# optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

if args.poison_type == 'TaCT' or args.poison_type == 'SleeperAgent':
    source_classes = config.source_class
else:
    source_classes = None

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch

    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]

    return [torch.cat(v, dim=0) for v in xy]

def mixmatch_train(
    model, xloader, uloader, criterion, optimizer, epoch, train_iteration, temperature, alpha, num_classes
):
    loss_meter = AverageMeter("loss")
    xloss_meter = AverageMeter("xloss")
    uloss_meter = AverageMeter("uloss")
    lambda_u_meter = AverageMeter("lambda_u")
    meter_list = [loss_meter, xloss_meter, uloss_meter, lambda_u_meter]

    xiter = iter(xloader)
    uiter = iter(uloader)

    model.train()
    gpu = next(model.parameters()).device
    start = time.time()
    for batch_idx in range(train_iteration):
        #print("batch_idx: ", batch_idx)
        try:

            xinput, xtarget = next(xiter)
            # xbatch = next(xiter)
            # xinput, xtarget = xbatch["img"], xbatch["target"]
        except:
            xiter = iter(xloader)
            xinput, xtarget = next(xiter)

            # xbatch = next(xiter)
            # xinput, xtarget = xbatch["img"], xbatch["target"]

        try:
            uinput1, uinput2 = next(uiter)
            #uinput2 = uinput1.clone().detach()

            # ubatch = next(uiter)
            # uinput1, uinput2 = ubatch["img1"], ubatch["img2"]
        except:
            uiter = iter(uloader)
            uinput1, uinput2 = next(uiter)
            #uinput2 = uinput1.clone().detach()


            # ubatch = next(uiter)
            # uinput1, uinput2 = ubatch["img1"], ubatch["img2"]

        batch_size = xinput.size(0)
        xtarget = torch.zeros(batch_size, num_classes).scatter_(
            1, xtarget.view(-1, 1).long(), 1
        )
        xinput = xinput.cuda(gpu, non_blocking=True)
        xtarget = xtarget.cuda(gpu, non_blocking=True)
        uinput1 = uinput1.cuda(gpu, non_blocking=True)
        uinput2 = uinput2.cuda(gpu, non_blocking=True)

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            uoutput1 = model(uinput1)
            uoutput2 = model(uinput2)
            p = (torch.softmax(uoutput1, dim=1) + torch.softmax(uoutput2, dim=1)) / 2
            pt = p ** (1 / temperature)
            utarget = pt / pt.sum(dim=1, keepdim=True)
            utarget = utarget.detach()

        # mixup
        all_input = torch.cat([xinput, uinput1, uinput2], dim=0)
        all_target = torch.cat([xtarget, utarget, utarget], dim=0)
        l = np.random.beta(alpha, alpha)
        l = max(l, 1 - l)
        idx = torch.randperm(all_input.size(0))
        input_a, input_b = all_input, all_input[idx]
        target_a, target_b = all_target, all_target[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabeled samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logit = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logit.append(model(input))

        # put interleaved samples back
        logit = interleave(logit, batch_size)
        xlogit = logit[0]
        ulogit = torch.cat(logit[1:], dim=0)

        Lx, Lu, lambda_u = criterion(
            xlogit,
            mixed_target[:batch_size],
            ulogit,
            mixed_target[batch_size:],
            epoch + batch_idx / train_iteration,
        )
        loss = Lx + lambda_u * Lu
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ema_optimizer.step()

        loss_meter.update(loss.item())
        xloss_meter.update(Lx.item())
        uloss_meter.update(Lu.item())
        lambda_u_meter.update(lambda_u)
        tabulate_step_meter(batch_idx, train_iteration, 3, meter_list)

    print("MixMatch training summary:")
    tabulate_epoch_meter(time.time() - start, meter_list)
    result = {m.name: m.total_avg for m in meter_list}

    return result


def poison_linear_record(model, loader, criterion):

    model.eval()
    num_data = len(loader.dataset)

    all_item_loss = []
    for data, target in tqdm(loader):
        data = data.cuda()
        target = target.cuda()

        data, target = data.cuda(), target.cuda()

        # with autocast():
        output = model(data)
        loss = criterion(output, target)

        loss = loss.squeeze().unsqueeze(0)

        loss = loss.tolist()[0]
        all_item_loss.extend(loss)

        data.cpu()
        target.cpu()

    return all_item_loss


def class_aware_loss_guided_split(all_item_loss, lists_overall_indices, choice_num, has_indices):

    indices_list = []
    for indices in lists_overall_indices:
        loss_list = [all_item_loss[i] for i in indices]
        sorted_overall_indices = sorted(zip(loss_list, indices))
        sorted_overall_indices = [item[1] for item in sorted_overall_indices]

        sorted_overall_indices = [x for x in sorted_overall_indices if x not in has_indices]
        filter_count = choice_num

        first_filter_indices = sorted_overall_indices[:filter_count]

        indices_list.extend(first_filter_indices)

    indices_list.extend(has_indices)

    return indices_list


def class_agnostic_loss_guided_split(all_item_loss, all_item_indices, epsilon = 0.5):

    sorted_overall_indices = sorted(zip(all_item_loss, all_item_indices))
    sorted_overall_indices = [item[1] for item in sorted_overall_indices]

    filter_count = int(len(all_item_loss) * epsilon)

    first_filter_indices = sorted_overall_indices[:filter_count]

    return first_filter_indices

def meta_split(all_item_loss, meta_all_item_loss, all_item_indices, epsilon = 0.5):

    all_item_loss = [x - y for x, y in zip(all_item_loss, meta_all_item_loss)]

    sorted_overall_indices = sorted(zip(all_item_loss, all_item_indices))
    sorted_overall_indices = [item[1] for item in sorted_overall_indices]

    filter_count = int(len(all_item_loss) * epsilon)

    first_filter_indices = sorted_overall_indices[:filter_count]

    return first_filter_indices





def train_the_virtual_model(meta_virtual_model, poison_train_loader, meta_optimizer, meta_criterion):
    """Train the virtual model in meta-split.
    """
    meta_virtual_model.train()
    for data, target in tqdm(poison_train_loader):
        data = data.cuda()
        target = target.cuda()

        meta_optimizer.zero_grad()
        output = meta_virtual_model(data)
        meta_criterion.reduction = "mean"
        loss = meta_criterion(output, target)

        loss.backward()
        meta_optimizer.step()





# 模型准备
resnet = arch(num_classes=num_classes)
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = SimCLR(backbone)
model.projection_head = nn.Linear(512, 43)
model.cuda()

 # 建立10个空列表
# lists 用来存储每一类的数据
lists = [[] for _ in range(num_classes)]

# 用来存放每一类数据在全局层面上的索引
lists_overall_indices = [[] for _ in range(num_classes)]

# 用来存放每一类数据所对应的loss值

lists_overall_loss = [[] for _ in range(num_classes)]

for i, (image, label) in enumerate(zip(poisoned_set_static.images, poisoned_set_static.gt)):
    # lists中对应的类列表添加图像
    lists[label].append(image.clone().detach())
    lists_overall_indices[label].append(i)

print("10类数据分配完毕")

# 准备初始化clean_data, 每一类准备10个数据
poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
poison_indices = torch.load(poison_indices_path)

clean_data_list = []
for indices in lists_overall_indices:
    candidate_set = set(indices) - (set(indices) & set(poison_indices))
    random_10 = random.sample(list(candidate_set), 10)
    random_10_list = list(random_10)
    clean_data_list.extend(random_10_list)

has_indices = clean_data_list.copy()

# 各项准备工作
criterion = nn.CrossEntropyLoss(reduction= "mean").cuda()
split_criterion = SCELoss(num_classes = num_classes, reduction="none").cuda()
semi_criterion = MixMatchLoss(rampup_length=120, lambda_u=15).cuda()
optimizer = torch.optim.Adam(model.parameters(), 0.002)
all_item_indices = list(range(len(poisoned_set.gt)))

choice_num = 0
import time

st = time.time()
for epoch in range(120):
    start_time = time.perf_counter()
    if epoch < 60:

        all_item_loss = poison_linear_record(model,poisoned_set_static_loader, split_criterion)

        if epoch % 5 == 0 and epoch != 0:
            choice_num += 10

        split_indices = class_aware_loss_guided_split(all_item_loss, lists_overall_indices, choice_num, has_indices)
        split_indices.sort()

        split_last_indices = [x for x in all_item_indices if x not in split_indices]
        split_last_indices.sort()

        xdata_set = torch.utils.data.Subset(poisoned_set, split_indices)
        udata_set = torch.utils.data.Subset(poisoned_set, split_last_indices)

    elif epoch < 90:

        all_item_loss = poison_linear_record(model, poisoned_set_static_loader, split_criterion)

        split_indices = class_agnostic_loss_guided_split(all_item_loss, all_item_indices, epsilon = 0.5)

        split_indices.sort()

        split_last_indices = [x for x in all_item_indices if x not in split_indices]
        split_last_indices.sort()

        xdata_set = torch.utils.data.Subset(poisoned_set, split_indices)
        udata_set = torch.utils.data.Subset(poisoned_set, split_last_indices)

    elif epoch < 120:

        all_item_loss = poison_linear_record(model, poisoned_set_static_loader, split_criterion)

        meta_virtual_model = deepcopy(model)

        children = list(meta_virtual_model.backbone.children())

        layer3 = children[4]
        layer4 = children[5]

        param_meta = [
            {'params': layer3.parameters()},
            {'params': layer4.parameters()},
            {'params': meta_virtual_model.projection_head.parameters()}
        ]

        meta_optimizer = torch.optim.Adam(param_meta, 0.015)
        meta_criterion = nn.CrossEntropyLoss(reduction="mean").cuda()

        for _ in range(1):
            train_the_virtual_model(
                meta_virtual_model=meta_virtual_model,
                poison_train_loader=poisoned_set_loader,
                meta_optimizer=meta_optimizer,
                meta_criterion=meta_criterion,
            )

        meta_all_item_loss = poison_linear_record(meta_virtual_model, poisoned_set_static_loader, split_criterion)

        split_indices = meta_split(all_item_loss, meta_all_item_loss, all_item_indices, epsilon = 0.5)

        split_indices.sort()

        split_last_indices = [x for x in all_item_indices if x not in split_indices]
        split_last_indices.sort()

        xdata_set = torch.utils.data.Subset(poisoned_set, split_indices)
        udata_set = torch.utils.data.Subset(poisoned_set, split_last_indices)

    print("干净池大小：", len(split_indices))
    print("干净候选池中有多少中毒样本：", len(set(split_indices) & set(poison_indices)))
    udata_set = TwiceAugmentedDataset(udata_set)

    xloader = torch.utils.data.DataLoader(
        xdata_set,
        batch_size=64, shuffle=True, drop_last=True, worker_init_fn=tools.worker_init, **kwargs)

    uloader = torch.utils.data.DataLoader(
        udata_set,
        batch_size=64, shuffle=True, drop_last=True, worker_init_fn=tools.worker_init, **kwargs)




    print("Epoch: ", epoch)

    poison_train_result = mixmatch_train(
        model, xloader, uloader, semi_criterion, optimizer, epoch, train_iteration = 1024, temperature = 0.5,alpha = 0.75,num_classes = 43
    )
    print(poison_train_result)

    # Test

    if args.dataset != 'ember':
        if True:
            # if epoch % 5 == 0:
            if args.dataset == 'imagenet':
                tools.test_imagenet(model=model, test_loader=test_set_loader,
                                    test_backdoor_loader=test_set_backdoor_loader)

            else:
                tools.test(model=model, test_loader=test_set_loader, poison_test=True,
                           poison_transform=poison_transform, num_classes=num_classes, source_classes=source_classes,
                           all_to_all=all_to_all)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print('Time: {:.2f}s'.format(elapsed_time) )


