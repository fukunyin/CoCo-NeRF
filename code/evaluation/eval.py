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
        self.batch_size = kwargs['batch_size']
        self.iter = kwargs['iter']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']

        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_string('dataset.scan_id', default=-1)
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

       
        is_continue = kwargs['is_continue']

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

        self.test_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf, is_train=False)

        self.ds_len = len(self.test_dataset)
        print('Finish loading test data. Data-set size: {0}'.format(self.ds_len))


        self.plot_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=False,
                                                           num_workers=0,
                                                           collate_fn=self.test_dataset.collate_fn
                                                           )

        # define model
        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model.geometric_init()


        self.do_vis = kwargs['do_vis']

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']


        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.test_dataset.total_pixels
        self.img_res = self.test_dataset.img_res
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=1000)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')


    def run(self):
        logging.basicConfig(level=logging.DEBUG,
                            filename=self.plots_dir + '/output.log',
                            datefmt='%Y/%m/%d %H:%M:%S',
                            format='%(asctime)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
        logger = logging.getLogger(__name__)

        print("testing...")
        self.model.eval()
        total_time = 0

        for data_index, (indices, model_input, ground_truth) in enumerate(self.plot_dataloader):

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

            plotpath = os.path.join(self.plots_dir, str(self.start_epoch))
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
            
        pertime = total_time / len(self.plot_dataloader)
        logger.info('-{:0>6}-   times:{:.4f}'.format(self.start_epoch, pertime))


            


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
