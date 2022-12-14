import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
import torch.utils.data
import time
from tqdm import tqdm
import numpy as np

from model.psnr import PSNR
from model.ssim import MS_SSIM

import vol_utils
import vol_utils.general as utils
import vol_utils.plots as plt
from vol_utils import rend_util

import logging


class VolSDFTrainRunner(object):
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        #torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])

        is_continue = kwargs["is_continue"]
        self.batch_size = kwargs['batch_size']
        self.iter = kwargs['iter']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']

        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_string('dataset.scan_id', default=-1)
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

        utils.mkdir_ifnotexists(os.path.join('../', self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        utils.mkdir_ifnotexists(os.path.join(self.expdir))

        self.plots_dir = os.path.join(self.expdir, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

        os.system(
            """cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, 'runconf.conf')))

        if not self.GPU_INDEX == 'ignore':
            os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf, is_train=True)
        self.test_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf, is_train=False)

        self.ds_len = len(self.train_dataset)
        print('Finish loading data. Data-set size: {0}'.format(self.ds_len))

        self.nepochs = int(self.iter / self.ds_len)
        print('RUNNING FOR {0}'.format(self.nepochs))

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            num_workers=4,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           num_workers=4,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        # define model
        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)
        print(self.model)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model.geometric_init()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        self.lr = self.conf.get_float('train.learning_rate')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Exponential learning rate scheduler
        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = self.nepochs * len(self.train_dataset)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1. / decay_steps))

        self.do_vis = kwargs['do_vis']

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=1000)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

    def run(self):
        logging.basicConfig(level=logging.DEBUG,
                            filename=self.plots_dir + '/output.log',
                            datefmt='%Y/%m/%d %H:%M:%S',
                            format='%(asctime)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
        logger = logging.getLogger(__name__)

        print("training...")
        for epoch in range(self.start_epoch, self.nepochs + 1):

            if epoch % self.checkpoint_freq == 0:
                self.save_checkpoints(epoch)

            if self.do_vis and epoch % self.plot_freq == 0 and epoch > 0 or epoch == 20:
                self.model.eval()
                total_time = 0

                for data_index, (indices, model_input, ground_truth) in enumerate(self.plot_dataloader):
                    self.train_dataset.change_sampling_idx(-1)
                    
                    model_input["intrinsics"] = model_input["intrinsics"].cuda()
                    model_input["uv"] = model_input["uv"].cuda()
                    model_input['pose'] = model_input['pose'].cuda()

                    split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)

                    val_start_time = time.time()

                    res = []
                    for s in tqdm(split):
                        out = self.model(s)
                        d = {'rgb_values': out['rgb_values'].detach(),
                             'normal_map': out['normal_map'].detach()}
                        res.append(d)
                    batch_size = ground_truth['rgb'].shape[0]

                    model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

                    used_time = time.time() - val_start_time
                    total_time = total_time + used_time
                    plot_data = self.get_plot_data(model_outputs, model_input['pose'], ground_truth['rgb'])

                    plotpath = os.path.join(self.plots_dir, str(epoch))
                    if not os.path.exists(plotpath):
                        os.makedirs(plotpath)

                    plt.plot(self.model.implicit_network,
                             indices,
                             plot_data,
                             plotpath,
                             data_index,
                             self.img_res,
                             **self.plot_conf
                             )
                    
                    break
                    

                pertime = total_time / len(self.plot_dataloader)
                logger.info('-{:0>6}-   times:{:.4f}'.format(epoch, pertime))

                self.model.train()
            

            self.train_dataset.change_sampling_idx(self.num_pixels)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()

                model_outputs = self.model(model_input)
                loss_output = self.loss(model_outputs, ground_truth)

                loss = loss_output['loss']

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                          ground_truth['rgb'].cuda().reshape(-1, 3))
                print(
                    '{0} [{1}] ({2}/{3}): loss = {4}, rgb_loss = {5}, eikonal_loss = {6}, psnr = {7}'
                        .format(self.expname, epoch, data_index, self.n_batches, loss.item(),
                                loss_output['rgb_loss'].item(),
                                loss_output['eikonal_loss'].item(),
                                psnr.item()))

                self.train_dataset.change_sampling_idx(self.num_pixels)
                self.scheduler.step()

        self.save_checkpoints(epoch)

    def get_plot_data(self, model_outputs, pose, rgb_gt):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].view((batch_size, num_samples, 3))
        normal_map = model_outputs['normal_map'].view((batch_size, num_samples, 3))
        normal_map = (normal_map + 1.) / 2.

        plot_data = {
            'rgb_gt': rgb_gt,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'normal_map': normal_map,
        }

        return plot_data
