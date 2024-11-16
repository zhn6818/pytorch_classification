import os
import tqdm
import time
import shutil

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn import functional as F
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils import init_logger, AverageMeter
from utils import get_scheduler, parser
from dataset import ClsDataset, train_transform, val_transform
from cls_models import ClsModel

def train(device, args):
    check_rootfolders()
    logger = init_logger(log_file=args.output + '/log')

    val_dataset = ClsDataset(
        list_file=args.val_list,
        transform=val_transform(size=args.input_size)
    )

    train_dataset = ClsDataset(
        list_file=args.train_list,
        transform=train_transform(size=args.input_size)
    )

    logger.info(f"Num train examples = {len(train_dataset)}")
    logger.info(f"Num val examples = {len(val_dataset)}")

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    criterion = torch.nn.CrossEntropyLoss().to(device)

    model = ClsModel(args.model_name, args.num_classes, args.is_pretrained)
    print(model.base_model)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    cudnn.benchmark = True

    for k, v in sorted(vars(args).items()):
        logger.info(f'{k} = {v}')

    for epoch in range(args.start_epoch, args.epochs):
        losses = AverageMeter()
        model.train()
        for step, (img, target, _) in enumerate(train_loader):
            img = img.to(device)
            target = target.to(device)

            output = model(img)
            loss = criterion(output, target)
            loss.backward()
            losses.update(loss.item(), img.size(0))

            if step % args.print_freq == 0:
                logger.info(f"Epoch: [{epoch}/{args.epochs}][{step}/{len(train_loader)}], lr: {optimizer.param_groups[-1]['lr']:.5f} \t loss = {losses.val:.4f}({losses.avg:.4f})")

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                preds, labels = [], []
                eval_pbar = tqdm.tqdm(val_loader, desc=f'epoch {epoch + 1} / {args.epochs} evaluating')
                for step, (img, target, _) in enumerate(eval_pbar):
                    img = img.to(device)
                    target = target.to(device)

                    output = model(img)
                    predict = torch.max(output, dim=1)[1]

                    labels.append(target)
                    preds.append(predict)

                labels = torch.cat(labels, dim=0).cpu().numpy()
                predicts = torch.cat(preds, dim=0).cpu().numpy()

                eval_result = (np.sum(labels == predicts)) / len(labels)
                logger.info(f'precision = {eval_result:.4f}')
                save_path = os.path.join(args.output, f'precision_{eval_result:.4f}_num_{epoch+1}')
                os.makedirs(save_path, exist_ok=True)
                model_to_save = model
                torch.save(model_to_save.state_dict(), os.path.join(save_path, f'epoch_{epoch+1}.pth'))

def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.output]
    for folder in folders_util:
        os.makedirs(folder, exist_ok=True)

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(args.train_list)
    print(f"Using device: {device}")
    train(device, args)