from core.engine import BaseEngine
from pyhocon import ConfigTree
from core.dataloaders import DataLoaderFactory
from core.models import ModelFactory
from Torchpie.torchpie.environment import experiment_path
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn, optim
from Torchpie.torchpie.utils.checkpoint import save_checkpoint
from Torchpie.torchpie.meters import AverageMeter
from Torchpie.torchpie.logging import logger
import time
from core.dataloaders.youtube_dataset import YoutubeDataset
from core.criterion import SmoothCrossEntropyLoss
from core.optimizer import CustomSchedule
from core.metrics import compute_epiano_accuracy
from pprint import pprint
import numpy as np
import argparse

import pdb


class Engine(BaseEngine):

    def __init__(self, cfg: ConfigTree, args):
            
        #pdb.set_trace()
        
        
        self.cfg = cfg
        self.device = self.cfg.get_string('device')
        self.summary_writer = SummaryWriter(log_dir=experiment_path)
        self.model_builder = ModelFactory(cfg)
        self.dataset_builder = DataLoaderFactory(cfg)

        self.train_ds = self.dataset_builder.build(split='train')
        self.test_ds = self.dataset_builder.build(split='val')
        self.ds: YoutubeDataset = self.train_ds.dataset

        
        
        self.train_criterion = nn.CrossEntropyLoss(
            ignore_index=self.ds.PAD_IDX
        )
        self.val_criterion = nn.CrossEntropyLoss(
            ignore_index=self.ds.PAD_IDX
        )
        
        
        
        
        
        self.model: nn.Module = self.model_builder.build(device=torch.device(self.device), wrapper=nn.DataParallel)
        
                
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)
        
        """
        self.optimizer = CustomSchedule(
            self.cfg.get_int('model.emb_dim'),
            optimizer=optimizer,
        )
        """
        self.num_epochs = cfg.get_int('num_epochs')
        
        if(args.load):
            print("loading model...")
            checkpoint = torch.load(experiment_path + '/checkpoint.pth.tar')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            self.num_epochs -= epoch
            self.epochs_left = epoch
            #self.loss = checkpoint['loss']
        else:
            self.epochs_left = 0
            #checkpoint = torch.load('../Foley-Music/exps/urmp-vn/checkpoint.pth.tar') #Transfer learning
            #self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            

        logger.info(f'Use control: {self.ds.use_control}')
        self.duration = self.cfg.get_float('dataset.duration')

    def train(self, epoch=0):
        loss_meter = AverageMeter('Loss')
        acc_meter = AverageMeter('Acc')
        num_iters = len(self.train_ds)
        self.model.train()
        count = 0
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        count = sum([np.prod(p.size()) for p in model_parameters])
        print(count)
        
        for i, data in enumerate(self.train_ds):
            midi_x, midi_y = data['midi_x'], data['midi_y']
            
            #pdb.set_trace()
            if self.ds.use_pose:
                feat = data['pose']
            elif self.ds.use_rgb:
                feat = data['rgb']
            elif self.ds.use_flow:
                feat = data['flow']
            elif self.ds.use_imu:
                feat = data['imu']
            else:
                raise Exception('No feature!')

            if(self.device == 'cuda'):
                feat, midi_x, midi_y = (
                        feat.cuda(non_blocking=True),
                        midi_x.cuda(non_blocking=True),
                        midi_y.cuda(non_blocking=True)
                )
           
            
            if self.ds.use_control:
                control = data['control']
                control = control.cuda(non_blocking=True)
            else:
                control = None

            output = self.model(feat, midi_x, pad_idx=self.ds.PAD_IDX, control=control)

            loss = self.train_criterion(output.view(-1, output.shape[-1]), midi_y.flatten())

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            acc = compute_epiano_accuracy(output, midi_y, pad_idx=self.ds.PAD_IDX)

            batch_size = len(midi_x)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc.item(), batch_size)

            logger.info(
                f'Train [{epoch}]/{self.num_epochs}][{i}/{num_iters}]\t'
                f'{loss_meter}\t{acc_meter}'
            )
        self.summary_writer.add_scalar('train/loss', loss_meter.avg, epoch)
        self.summary_writer.add_scalar('train/acc', acc_meter.avg, epoch)
        return loss_meter.avg

    def test(self, epoch=0):
        loss_meter = AverageMeter('Loss')
        acc_meter = AverageMeter('Acc')
        num_iters = len(self.test_ds)
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(self.test_ds):
                midi_x, midi_y = data['midi_x'], data['midi_y']
                #pdb.set_trace()
                if self.ds.use_pose:
                    feat = data['pose']
                elif self.ds.use_rgb:
                    feat = data['rgb']
                elif self.ds.use_flow:
                    feat = data['flow']
                elif self.ds.use_imu:
                    feat = data['imu']
                else:
                    raise Exception('No feature!')

                if(self.device == 'cuda'):
                        feat, midi_x, midi_y = (
                        feat.cuda(non_blocking=True),
                        midi_x.cuda(non_blocking=True),
                        midi_y.cuda(non_blocking=True)
                        )
                
                if self.ds.use_control:
                    control = data['control']
                    control = control.cuda(non_blocking=True)
                else:
                    control = None

                output = self.model(feat, midi_x, pad_idx=self.ds.PAD_IDX, control=control)

                """
                For CrossEntropy
                output: [B, T, D] -> [BT, D]
                target: [B, T] -> [BT]
                """
                loss = self.val_criterion(output.view(-1, output.shape[-1]), midi_y.flatten())

                acc = compute_epiano_accuracy(output, midi_y)

                batch_size = len(midi_x)
                loss_meter.update(loss.item(), batch_size)
                acc_meter.update(acc.item(), batch_size)
                logger.info(
                    f'Val [{epoch}]/{self.num_epochs}][{i}/{num_iters}]\t'
                    f'{loss_meter}\t{acc_meter}'
                )
            self.summary_writer.add_scalar('val/loss', loss_meter.avg, epoch)
            self.summary_writer.add_scalar('val/acc', acc_meter.avg, epoch)

        return loss_meter.avg

    @staticmethod
    def epoch_time(start_time: float, end_time: float):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def run(self):
        best_loss = float('inf')
        for epoch in range(self.num_epochs):
            start_time = time.time()
            _train_loss = self.train(epoch)
            loss = self.test(epoch)
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            logger.info(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            
            torch.save({
            'epoch': epoch + self.epochs_left,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            }, experiment_path + '/' + 'checkpoint.pth.tar')

    def close(self):
        self.summary_writer.close()


def main(args):
    from torchpie.config import config as cfg
    print('=' * 100)
    pprint(cfg)
    print('=' * 100)
    engine = Engine(cfg, args)
    engine.run()
    engine.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l', '--load', action='store_true')
    args, unknown = parser.parse_known_args()
    main(args)
