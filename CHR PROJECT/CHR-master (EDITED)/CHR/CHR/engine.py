import os
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import torchtools as tnt
from util import AveragePrecisionMeter, Warp


class Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('train_image_size') is None:
            self.state['train_image_size'] = 256
        if self._state('test_image_size') is None:
            self.state['test_image_size'] = 224

        if self._state('batch_size') is None:
            self.state['batch_size'] = 64

        if self._state('train_workers') is None:
            self.state['train_workers'] = 16
        if self._state('test_workers') is None:
            self.state['test_workers'] = 4

        if self._state('multi_gpu') is None:
            self.state['multi_gpu'] = False

        if self._state('device_ids') is None:
            self.state['device_ids'] = [0, 1, 2, 3]

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []

        # meters
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()
        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = True
        if self._state('print_freq') is None:
            self.state['print_freq'] = 0

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        loss = self.state['meter_loss'].value()[0]
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}'.format(self.state['epoch'], loss=loss))
            else:
                print('Test: \t Loss {loss:.4f}'.format(loss=loss))
        return loss

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        # record loss
        self.state['loss_batch'] = self.state['loss'].data
        self.state['meter_loss'].add(self.state['loss_batch'].cpu().numpy())

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'


                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

    def run(self, training, model, criterion, data_loader, optimizer=None):
        """
        Run the training or evaluation.
        """
        if training:
            model.train()
        else:
            model.eval()

        self.on_start_epoch(training, model, criterion, data_loader, optimizer)
        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            self.state['iteration'] = i

            self.state['data_time'].update(time.time() - end)
            if self._state('use_gpu'):
                input, target = input.cuda(), target.cuda()

            self.on_start_batch(training, model, criterion, data_loader, optimizer)
            output = model(input)
            if isinstance(output, tuple):
                output = output[0]

            self.state['loss'] = criterion(output, target)
            if training:
                optimizer.zero_grad()
                self.state['loss'].backward()
                optimizer.step()

            self.on_end_batch(training, model, criterion, data_loader, optimizer)
            self.state['batch_time'].update(time.time() - end)
            end = time.time()

        return self.on_end_epoch(training, model, criterion, data_loader, optimizer)