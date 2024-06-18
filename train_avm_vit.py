from multiprocessing import reduction
import os
import argparse
import builtins
import random
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import skimage.io
from skimage.measure import find_contours
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import multiprocessing as mp
import torch.distributed as dist
import utils
import torch.backends.cudnn as cudnn
from torchvision import transforms
import cv2
from datasets import AVE, VGGSound, CremadDataset
from models import LAVISH, Shared_Transformer, Audio_Transformer, Image_Transformer, \
    CLS_Token_Proj_AVIT, CNT_Token_Proj_AVIT, Mean_Pooled_Proj_AVIT, \
        Attn_Pooled_Proj_AVIT, MIM_Proj_AVIT, MA_AVT
import soundfile as sf
from vis import display_instances, magnitude2heatmap, HTMLVisualizer

from arguments import ArgParser
import warnings

best_acc = -1.

def main(args):
    args.id += '-{}-{}'.format(args.vis_encoder_type, args.vit_type)
    args.id += '-MT_{}-UT_{}'.format(int(args.multimodal_token), int(args.unimodal_token))
    args.id += '-LSA_{}'.format(int(args.LSA))
    args.id += '-LAV_{}'.format(int(args.lavish_adapter))
    args.id += '-GM_{}'.format(int(args.grad_mod))
    args.id += '-BG_{}'.format(int(args.bg_cls))
    args.id += '-CNT_{}'.format(args.contrastive)
    args.id+= '-{}'.format(args.dataset)
    args.id += '-epoch{}'.format(args.epochs)
    args.id += '-batch{}'.format(args.batch_size)
    args.id += '-lr{}'.format(args.lr)
    args.id += '-step{}'.format(args.lr_step)
    args.id += '-seed{}'.format(args.seed)

    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    args.output_dir = os.path.join(args.output_dir, args.id)

    args.vis = os.path.join(args.output_dir, 'visualization')
    args.ckpt = os.path.join(args.output_dir, "checkpoints")
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    if not os.path.isdir(args.vis):
        os.makedirs(args.vis)
        
        if not os.path.isdir(os.path.join(args.vis, "val")):
            os.makedirs(os.path.join(args.vis, "val"))
            
        if not os.path.isdir(os.path.join(args.vis, "test")):
            os.makedirs(os.path.join(args.vis, "test"))
    
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)
        
    args.log_fn = f"{args.output_dir}/train.log"
    
    if os.path.isfile(args.log_fn):
        os.remove(args.log_fn)

    # Create model dir
    utils1.save_json(vars(args), os.path.join(args.output_dir, 'configs.json'), sort_keys=True, save_pretty=True)

    mp.set_start_method('spawn')
    args.dist_url = f'tcp://{args.node}:{args.port}'
    print('Using url {}'.format(args.dist_url))

    ngpus_per_node = args.ngpu if args.ngpu else torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))

    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # Setup distributed environment
    if args.multiprocessing_distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    if args.gpu is not None:
        device = torch.device('cuda:{}'.format(args.gpu))

    def print_and_log(*content, **kwargs):
        # suppress printing if not first GPU on each node
        if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
            return
        msg = ' '.join([str(ct) for ct in content])
        sys.stdout.write(msg+'\n')
        sys.stdout.flush()
        with open(args.log_fn, 'a') as f:
            f.write(msg+'\n')

    builtins.print = print_and_log

    print("Loading model......")

    if args.model == "shared_transformer":
        model = Shared_Transformer(args)
    if args.model == "LAVISH":
        model = LAVISH(args)
    elif args.model == "audio_transformer":
        model = Audio_Transformer(args)
    elif args.model == "image_transformer":
        model = Image_Transformer(args)
    elif args.model == "audio_transformer":
        model = Audio_Transformer(args)
    elif args.model == "cls_token_proj_avit":
        model = CLS_Token_Proj_AVIT(args)
    elif args.model == "cnt_token_proj_avit":
        model = CNT_Token_Proj_AVIT(args)
    elif args.model == "mean_pooled_proj_avit":
        model = Mean_Pooled_Proj_AVIT(args)
    elif args.model == "attn_pooled_proj_avit":
        model = Attn_Pooled_Proj_AVIT(args)
    elif args.model == "mim_proj_avit":
        model = MIM_Proj_AVIT(args)
    elif args.model == "ma_avt":
        model = MA_AVT(args)
    else:
        print("Not loaded")


    print("Model is loaded!")

    if not args.full_tune:   
        for name, param in model.named_parameters():            
            if 'cls_token' in name:
                param.requires_grad = True
            elif 'ViT'in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
 
    ## Count paramters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    token_params = sum(p.numel() for name, p in model.named_parameters() if 'tokens' in name)
    infer_trained_params = sum(p.numel() for name, p in model.named_parameters() if ('tokens' in name or 'audio_attn' in name or 'vis_attn' in name or 'shared_attn' in name))


    print(f"Total Params: {total_params/1000000: 6.4f} M, or {total_params/1000000: .4e} M")
    print(f"Trainable Params: {trainable_params/1000000: 6.4f} M or {trainable_params/1000000: .4e} M")
    print(f"Token Params: {token_params/1000000: 6.4f} M or {token_params/1000000: .4e} M")
    print(f"Infer Trained Params: {infer_trained_params/1000000: 6.4f} M or {infer_trained_params/1000000: .4e} M")

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.multiprocessing_distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    
    print(model)

    # Optimizer
    optimizer, scheduler = utils1.build_optimizer_and_scheduler_adam(model, args)

    # History of peroformance
    history = {
        'train': {'epoch': [], 'loss': [], 'fg_loss': [], 'bg_loss': [], 'cnt_loss':[], 'fg_acc': [], 'bg_acc': [], 'total_acc': []},
        'val': {'epoch': [], 'loss': [], 'fg_loss': [], 'bg_loss': [], 'cnt_loss':[], 'fg_acc': [], 'bg_acc': [], 'total_acc': []}} 

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            history = checkpoint['history']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    torch.cuda.empty_cache()
    
    # Dataloaders
    if args.dataset == "ave":
        Dataset = AVE
    elif args.dataset == "vggsound":
        Dataset = VGGSound
    elif args.dataset == "cremad":
        Dataset = CremadDataset

    traindataset = Dataset(args, mode='train')
    train_sampler = None
    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(traindataset)
    train_loader = torch.utils.data.DataLoader(
        traindataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True,
        persistent_workers=args.workers > 0)

    valdataset = Dataset(args, mode='val')
    val_sampler = None
    if args.multiprocessing_distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(valdataset)
    val_loader = torch.utils.data.DataLoader(
        valdataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler, drop_last=False,
        persistent_workers=args.workers > 0)

    testdataset = Dataset(args, mode='test')
    test_sampler = None
    if args.multiprocessing_distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(testdataset)
    test_loader = torch.utils.data.DataLoader(
        testdataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=test_sampler, drop_last=False,
        persistent_workers=args.workers > 0)

    print("Loaded dataloader.")
    print(f"Size of Training dataset: {len(traindataset)}")
    print(f"Size of Validation dataset: {len(valdataset)}")
    print(f"Size of Test dataset: {len(testdataset)}")


    args.epoch_iters = len(train_loader)
    print('1 Epoch = {} iters'.format(args.epoch_iters))

    criterion = nn.CrossEntropyLoss()

    if args.mode == 'val':	
        assert args.resume is not None, "No pretrained model to run validation/test"
        acc = validate(val_loader, model, criterion, history, 0, device, args, prefix="Val") 
        print(f"Validation accuracy: {acc:6.3f}")

        # visualization(args, model, Dataset, mode='val')

        return 

    elif args.mode == 'test':
        assert args.resume is not None, "No pretrained model to run validation/test"
        acc = validate(test_loader, model, criterion, history, 0, device, args, prefix="Test")

        print(f"Test accuracy: {acc:6.3f}")
     
        # visualization(args, model, Dataset, mode='test')

        return 


    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.multiprocessing_distributed:
            train_loader.sampler.set_epoch(epoch)

        train(train_loader, model, optimizer, criterion, epoch, history, args)

        torch.cuda.empty_cache()

        if epoch % args.eval_epoch == 0:
            acc = validate(val_loader, model, criterion, history, epoch, device, args, prefix='Val')
        
            is_best = acc > best_acc
            best_acc = max(acc, best_acc)

            print(f'Accuracy (epoch {epoch}): {acc}')
            print(f'Best Val Accuracy: {best_acc}')

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                checkpoint = {
                    'epoch': epoch,
                    'arch': args.model,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict(),
                    'history' : history
                }
                save_checkpoint(checkpoint, history, epoch, is_best, device, args)
            
            torch.distributed.barrier()

        scheduler.step()


    print('Training Done!')
    print('Running Test Evalutation......')

    print("=> loading best checkpoint...")
    args.path = "{}/model_{}".format(args.ckpt, "best.pth.tar")

    if args.gpu is None:
        checkpoint = torch.load(args.path)
    elif torch.cuda.is_available():
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.path, map_location=loc)

    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])

    acc = validate(test_loader, model, criterion, history, epoch, device, args, prefix='Test')

    print(f"Best Val accuracy: {best_acc:0.2f}")
    print(f"Test accuracy: {acc:0.2f}")

    print('Test Evaluation Done!')

    # print("Running Test Visualization...")
    # visualization(args, model, Dataset, mode='test')
    # print("Test Visualization Done!")


def train(train_loader, model, optimizer, criterion, epoch, history, args):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_mtr = AverageMeter('Loss', ':.3f')
    fg_loss_mtr = AverageMeter('FG Loss', ':.3f')
    cnt_loss_mtr = AverageMeter('Contr Loss', ':.3f')
    fg_acc_mtr = AverageMeter('FG Acc', ':6.2f')

    bg_loss_mtr = AverageMeter('BG Loss', ':.3f')
    bg_acc_mtr = AverageMeter('BG Acc', ':6.2f')
    total_acc_mtr = AverageMeter('Total Acc', ':6.2f')

    if args.bg_cls:
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, loss_mtr, fg_loss_mtr, bg_loss_mtr, cnt_loss_mtr, fg_acc_mtr, bg_acc_mtr, total_acc_mtr],
            prefix="Epoch: [{}]".format(epoch)
        )
    else:
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, loss_mtr, fg_loss_mtr, cnt_loss_mtr, fg_acc_mtr],
            prefix="Epoch: [{}]".format(epoch)
        )

    end = time.time()
    for i, data in enumerate(train_loader):
        audio = data['audio_spec']
        target = data['target']
        image = data['image']

        data_time.update(time.time() - end)
        global_step = i + len(train_loader) * epoch
        utils1.adjust_learning_rate(optimizer, epoch + i / len(train_loader), args)

        if args.gpu is not None:
            audio_spec = audio.to(args.gpu, non_blocking=True)
            target = target.squeeze().to(args.gpu, non_blocking=True)
            image = image.to(args.gpu, non_blocking=True)

        outputs = model(audio_spec.float(), image.float(), target)
        
        loss, fg_loss, bg_loss, cnt_loss = outputs['loss'], outputs['fg_loss'], outputs['bg_loss'], outputs['cnt_loss']
        
        fg_acc = accuracy(outputs['p_fg'], target, args)
        if args.bg_cls:
            bg_acc = binary_accuracy(outputs['p_bg'], target, args)
            total_acc = total_accuracy(outputs['p_fg'], outputs['p_bg'], target, args)

        # measure accuracy and record loss
        loss_mtr.update(loss.item(), image.size(0))
        fg_loss_mtr.update(fg_loss.item(), image.size(0))
        if cnt_loss != 0:
            cnt_loss_mtr.update(cnt_loss.item(), image.size(0))

        if args.bg_cls:
            bg_loss_mtr.update(bg_loss.item(), image.size(0))
            fg_acc_mtr.update(fg_acc, image.size(0))
            bg_acc_mtr.update(bg_acc, image.size(0))
            total_acc_mtr.update((fg_acc*(target != args.bg_label).sum() + bg_acc*(target == args.bg_label).sum())/image.size(0) , image.size(0))

        optimizer.zero_grad()
        loss.backward()

        # gradient clip
        if args.clip_norm != 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)  # clip gradient

        if args.grad_mod:
            modulate_gradients(model, outputs['coeff_v'], outputs['coeff_a'])

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0 or i == len(train_loader) - 1:
            progress.display(i)

        fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
        history['train']['epoch'].append(fractional_epoch)
        history['train']['loss'].append(loss_mtr.avg)
        history['train']['fg_loss'].append(fg_loss_mtr.avg)
        history['train']['bg_loss'].append(bg_loss_mtr.avg)
        history['train']['cnt_loss'].append(cnt_loss_mtr.avg)
        history['train']['fg_acc'].append(fg_acc_mtr.avg)
        history['train']['bg_acc'].append(bg_acc_mtr.avg)
        history['train']['total_acc'].append(total_acc_mtr.avg)

        del loss

@torch.no_grad()
def validate(test_loader, model, criterion, history, epoch, device, args, prefix='Test'):
    model.train(False)
    # evaluator = utils.EvaluatorFull()

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    fg_losses = AverageMeter('FG Loss', ':6.3f')
    bg_losses = AverageMeter('BG Loss', ':6.3f')
    cnt_losses = AverageMeter('Cnt Loss', ':6.3f')
    fg_accs = AverageMeter('FG Acc', ':6.2f')
    bg_accs = AverageMeter('BG Acc', ':6.2f')
    total_accs = AverageMeter('Total Acc', ':6.2f')

    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, fg_losses, bg_losses, cnt_losses, fg_accs, bg_accs, total_accs],
        prefix='{}: '.format(prefix))

    end = time.time()

    for i, data in enumerate(test_loader):
        audio = data['audio_spec']
        target = data['target']
        image = data['image']

        if args.gpu is not None:
            audio_spec = audio.to(args.gpu, non_blocking=True)
            target = target.squeeze().to(args.gpu, non_blocking=True)
            image = image.to(args.gpu, non_blocking=True)

        # compute output
        outputs = model(audio_spec.float(), image.float(), target)
        loss, fg_loss, bg_loss, cnt_loss = outputs['loss'], outputs['fg_loss'], outputs['bg_loss'], outputs['cnt_loss']
        
        fg_acc = accuracy(outputs['p_fg'], target, args)
        if args.bg_cls:
            bg_acc = binary_accuracy(outputs['p_bg'], target, args)
            total_acc = total_accuracy(outputs['p_fg'], outputs['p_bg'], target, args)

        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))
        fg_losses.update(fg_loss.item(), image.size(0))

        if cnt_loss != 0:
            cnt_losses.update(cnt_loss.item(), image.size(0))

        if args.bg_cls:
            bg_losses.update(bg_loss.item(), image.size(0))
            fg_accs.update(fg_acc, image.size(0))
            bg_accs.update(bg_acc, image.size(0))
            total_accs.update((fg_acc*(target != args.bg_label).sum() + bg_acc*(target == args.bg_label).sum())/image.size(0) , image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0 or i == len(test_loader) - 1:
            progress.display(i)    

    history['val']['epoch'].append(epoch)
    history['val']['loss'].append(losses.avg)
    history['val']['fg_loss'].append(fg_losses.avg)
    history['val']['bg_loss'].append(bg_losses.avg)
    history['val']['cnt_loss'].append(cnt_losses.avg)
    history['val']['fg_acc'].append(fg_accs.avg)
    history['val']['bg_acc'].append(bg_accs.avg)
    history['val']['total_acc'].append(total_accs.avg)


    for mode in history.keys():
        for key in history[mode]:
            val = torch.tensor(history[mode][key], dtype=torch.float32, device=device)
            dist.all_reduce(val, dist.ReduceOp.SUM, async_op=False)
            val = (val / args.world_size).tolist()
            history[mode][key] = val
    
    if args.multiprocessing_distributed:
        fg_accs.all_reduce()
        bg_accs.all_reduce()
        total_accs.all_reduce()

    # plotting and saving
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % args.world_size == 0):

        # Plot figure
        if epoch > 0 and "test" not in args.vis:            
            print('Plotting figures...')
            plot_loss_metrics(args.output_dir, history, args)

    return total_accs.avg


def inverse_normalize(tensor):
    inverse_mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
    inverse_std = [1.0/0.229, 1.0/0.224, 1.0/0.225]
    tensor = transforms.Normalize(inverse_mean, inverse_std)(tensor)
    return tensor


@torch.no_grad()
def visualization(args, model, Dataset, mode='test'):
    model.train(False)

    args.vis_path = os.path.join(args.vis, mode)

    # initialize HTML header
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % args.world_size == 0):
        visualizer = HTMLVisualizer(os.path.join(args.vis_path, 'index.html'))
        header = ['Class Name', 'Input Audio', 'Input Image', 'Attention Map', 'Weighted Attention', 'Contour Map']

        visualizer.add_header(header)

    vis_rows = []
    
    # transform = transforms.Compose([
	# 			transforms.Resize([224, 224], interpolation=Image.BICUBIC),
	# 			transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),])

    dataset = Dataset(args, mode=mode, return_audio=True)
    idx_to_cls = dataset.idx_to_class
    # dataset.my_normalize = transform

    dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), args.num_vis, replace=False).tolist())

    sampler = None

    if args.multiprocessing_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=sampler, drop_last=False,
        persistent_workers=args.workers > 0)

    for step, data in enumerate(loader):
        audio = data['audio_spec']
        target = data['target']
        image = data['image']
        waveform = data['waveform']
        spectrogram = data['spectrogram']

        if args.gpu is not None:
            audio = audio.to(args.gpu, non_blocking=True)
            target = target.squeeze().to(args.gpu, non_blocking=True)
            image = image.to(args.gpu, non_blocking=True)
            waveform = data['waveform']
            spectrogram = data['spectrogram']

        # compute output
        outputs = model(audio.float(), image.float(), target)
        image, audio = image.squeeze(), audio.squeeze()

        vis_attn = outputs['vis_attn']

        w, h = image.shape[-1], image.shape[-2]
        num_tokens = vis_attn.shape[-1]
        patch_size = w // int(np.sqrt(num_tokens))
        
        B = audio.shape[0]
        
        # we keep only the output patch attention
        w_featmap = w // patch_size
        h_featmap = h // patch_size
        vis_attn = vis_attn.reshape(B, 1, w_featmap, h_featmap)

        # vis_attn = nn.functional.interpolate(vis_attn, scale_factor=patch_size, mode="nearest")
        # vis_attn = F.interpolate(vis_attn, size=(224, 224), mode='bicubic', align_corners=True)
        vis_attn = nn.functional.interpolate(vis_attn, scale_factor=patch_size, mode="nearest")
        vis_attn = vis_attn.data.cpu().numpy()
         
        for ind in range(B): 
            img = image[ind].squeeze()
            spec = spectrogram[ind]
            sound = waveform[ind]
            pred = vis_attn[ind].squeeze()

            class_name = idx_to_cls[target[ind].item()]

            number = B * args.gpu + ind + step * B * args.world_size

            path = f"Sample{number}_{class_name}"
            os.makedirs(os.path.join(args.vis_path, path), exist_ok=True)

            img_name =  os.path.join(path, f"image" + ".jpg")
            denorm_image = inverse_normalize(img[None])[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
            denorm_image = (denorm_image*255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.vis_path, img_name), denorm_image)

            attn_name = os.path.join(path, f"attn_map" + ".jpg")
            pred = vis_attn[ind].squeeze()
            # pred = pred - pred.min() 
            # pred = pred / pred.max()
            pred = np.uint8(pred*255)
            pred = cv2.applyColorMap(pred[:, :, np.newaxis], cv2.COLORMAP_JET)[:, :, ::-1]
            plt.imsave(fname=os.path.join(args.vis_path, attn_name), arr=pred, format='jpg')

            # cv2.imwrite(os.path.join(args.vis_path, attn_name), pred)

            weighted_name = os.path.join(path, f"weighted" + ".jpg")
            fin = cv2.addWeighted(pred, 0.8, np.uint8(denorm_image), 0.2, 0)
            cv2.imwrite(os.path.join(args.vis_path, weighted_name), fin)

            mag_amp = magnitude2heatmap(spec.cpu().numpy())
            filename_mag = os.path.join(path, f"spectrogram" + ".jpg")
            plt.imsave(os.path.join(args.vis_path, filename_mag), mag_amp[::-1, :, :])

            filename_wav =  os.path.join(path, f"audio" + ".wav")
            sf.write(os.path.join(args.vis_path, filename_wav), sound.cpu(), 16000)

            row_elements = [{'text': class_name},
                            {'image': filename_mag, 'audio': filename_wav},
                            {'image': img_name},
                            {'image': attn_name},
                            {'image': weighted_name}]

            vis_rows.append(row_elements)

    # aggregate all vis rows
    all_vis_rows = [None for _ in range(args.world_size)]
    dist.all_gather_object(all_vis_rows, vis_rows)
    vis_rows = [vis_row for vis_rows in all_vis_rows for vis_row in vis_rows]

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % args.world_size == 0):
        print('Plotting html for visualization...')

        visualizer.add_rows(vis_rows)
        visualizer.write_html()


def modulate_gradients(model, coeff_v, coeff_a):
    for name, parms in model.module.named_parameters():
        if 'audio' in name and parms.numel() != 1:
            parms.grad = parms.grad * coeff_a + \
                            torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)

        if 'vis' in name and parms.numel() != 1:
            parms.grad = parms.grad * coeff_v + \
                            torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)

def accuracy(output, target, args):
    """Computes the accuracy"""
    with torch.no_grad():
        batch_size = target.size(0)
        
        pred = torch.argmax(output, dim=1)
        
        if args.bg_cls:
            bg_label = (target == args.bg_label).long()
            mask = torch.where(bg_label == 0)[0]
            pred = pred[mask]
            target = target[mask]
        
        correct = pred.eq(target).sum()
        return correct * 100 / pred.size(0)

def total_accuracy(pred_fg, pred_bg, target, args):
    """Computes the accuracy"""
    with torch.no_grad():
        batch_size = target.size(0)
        
        pred_fg = torch.argmax(pred_fg, dim=1)
        pred_bg = (F.sigmoid(pred_bg) > 0.5).long()
        
        for i in range(batch_size):
            if pred_bg[i] == 0:
                pred_bg[i] = pred_fg[i]
            else:
                pred_bg[i] = args.bg_label

        correct = pred_bg.eq(target).sum()
        return correct * 100 / batch_size

def binary_accuracy(output, target, args):
    """Computes the binary accuracy"""
    with torch.no_grad():
        batch_size = target.size(0)

        pred = (F.sigmoid(output) > 0.5).long()

        bg_label = (target == args.bg_label).long()
        
        correct = pred.eq(bg_label).sum()
        return correct * 100 / batch_size

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", fp=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.fp = fp

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        msg = '\t'.join(entries)
        print(msg, flush=True)
        if self.fp is not None:
            self.fp.write(msg+'\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(checkpoint, history, epoch, is_best, device, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))

    suffix_latest = 'latest.pth.tar'
    suffix_best = 'best.pth.tar'
    
    # aggregate history
    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(checkpoint,
               '{}/model_{}'.format(args.ckpt, suffix_latest))

    if is_best:
        torch.save(checkpoint,
                '{}/model_{}'.format(args.ckpt, suffix_best))

def plot_loss_metrics(path, history, args):
    fig = plt.figure()
    plt.plot(history['train']['epoch'], history['train']['loss'],
             color='b', label='training loss')
    plt.plot(history['val']['epoch'], history['val']['loss'],
             color='c', label='validation loss')
    plt.legend()
    plt.title("Training vs Validation Loss")
    fig.savefig(os.path.join(path, 'loss.png'), dpi=200)
    plt.close('all')


    fig = plt.figure()

    loss = history['train']['fg_loss']
    plt.plot(history['train']['epoch'], loss,
             color='b', label='train_fg_loss')

    loss = history['train']['bg_loss']
    plt.plot(history['train']['epoch'], loss,
             color='lightskyblue', label='train_bg_loss')

    loss = history['train']['cnt_loss']
    plt.plot(history['train']['epoch'], loss,
             color='r', label='train_contrastive_loss')

    loss = history['val']['fg_loss']
    plt.plot(history['val']['epoch'], loss,
             color='c', label='val_fg_loss')

    loss = history['val']['bg_loss']
    plt.plot(history['val']['epoch'], loss,
             color='y', label='val_bg_loss')

    loss = history['val']['cnt_loss']
    plt.plot(history['val']['epoch'], loss,
             color='g', label='val_contrastive_loss')

    plt.legend()
    fig.savefig(os.path.join(path, f'loss_details.png'), dpi=200)
    plt.close('all')

    fig = plt.figure()
    plt.plot(history['train']['epoch'], history['train']['fg_acc'],
             color='b', label='Train FG ACC')
    plt.plot(history['train']['epoch'], history['train']['bg_acc'],
                color='r', label='Train BG ACC')
    plt.plot(history['train']['epoch'], history['train']['total_acc'],
                color='y', label='Train Total ACC')

    plt.plot(history['val']['epoch'], history['val']['total_acc'],
             color='c', label='Val Total ACC')

    plt.legend()
    plt.title("Training vs Validation Metrics")
    fig.savefig(os.path.join(path, 'metrics.png'), dpi=200)
    plt.close('all')


if __name__ == "__main__":
    parser = ArgParser()
    args = parser.parse()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main(args)
