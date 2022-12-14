import os
import torch
import numpy as np
import toml

from torchvision import transforms

import vol_utils.general as utils
from vol_utils import image_util
from vol_utils import rend_util

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id,
                 is_train,
                 num_views = 32
                 ):

        self.instance_dir = os.path.join(data_dir, scan_id)
 
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res


        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None

        image_dir = '{0}/image'.format(self.instance_dir)
        image_path = sorted(utils.glob_imgs(image_dir))

        self.n_images = len(image_path)
        
        # load split
        self._config = toml.load("../data/dtu/train_val_split.toml")
        train_list = set(self._config["scenes"][scan_id]["default_views_configs"][str(num_views)])
        test_list = set(range(self.n_images)) - train_list

        if is_train:
            self.list = train_list
        else:
            self.list = test_list
        
        if is_train == 'render':
            self.list = set(range(self.n_images))

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_alls = []
        self.pose_alls = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_alls.append(torch.from_numpy(intrinsics).float())
            self.pose_alls.append(torch.from_numpy(pose).float())

        image_paths = []
        self.intrinsics_all = []
        self.pose_all = []
        for idx in self.list:
            image_paths.append(image_path[idx])
            self.intrinsics_all.append(self.intrinsics_alls[idx])
            self.pose_all.append(self.pose_alls[idx])
        
        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
        
        
        self.n_images = len(self.rgb_images)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # [C, H, W], color range [0, 255] -> [0, 1]
                transforms.Resize([256, 256]),
                transforms.Normalize([0.5], [0.5]),  # color range [0, 1] -> [-1, 1]
            ]
        )

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)


        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']






class SceneDataset_h3ds(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id,
                 is_train,
                 num_views = 32
                 ):

        self.instance_dir = os.path.join(data_dir, scan_id)
 
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None

        image_dir = '{0}/image'.format(self.instance_dir)
        image_path = sorted(utils.glob_imgs(image_dir))

        self.n_images = len(image_path)
        
        # load split
        self._config = toml.load("../data/h3ds/train_val_split.toml")
        train_list = set(self._config["scenes"][scan_id]["default_views_configs"][str(num_views)])
        test_list = set(range(self.n_images)) - train_list

        if is_train:
            self.list = train_list
        else:
            self.list = test_list
            
        if is_train == 'render':
            self.list = set(range(self.n_images))

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_alls = []
        self.pose_alls = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_alls.append(torch.from_numpy(intrinsics).float())
            self.pose_alls.append(torch.from_numpy(pose).float())

        image_paths = []
        self.intrinsics_all = []
        self.pose_all = []
        for idx in self.list:
            image_paths.append(image_path[idx])
            self.intrinsics_all.append(self.intrinsics_alls[idx])
            self.pose_all.append(self.pose_alls[idx])

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
        
        
        self.n_images = len(self.rgb_images)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # [C, H, W], color range [0, 255] -> [0, 1]
                transforms.Resize([256, 256]),
                transforms.Normalize([0.5], [0.5]),  # color range [0, 1] -> [-1, 1]
            ]
        )

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']






class SceneDataset_bmvs(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id,
                 is_train,
                 num_views = 32
                 ):

        self.instance_dir = os.path.join(data_dir, scan_id)
 
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None

        image_dir = '{0}/image'.format(self.instance_dir)
        image_path = sorted(utils.glob_imgs(image_dir))

        self.n_images = len(image_path)
        
        # load split
        self._config = toml.load("../data/bmvs/train_val_split.toml")
        train_list = set(self._config["scenes"][scan_id]["default_views_configs"][str(num_views)])
        test_list = set(range(self.n_images)) - train_list

        if is_train:
            self.list = train_list
        else:
            self.list = test_list
        
        if is_train == 'render':
            self.list = set(range(self.n_images))

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_alls = []
        self.pose_alls = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_alls.append(torch.from_numpy(intrinsics).float())
            self.pose_alls.append(torch.from_numpy(pose).float())

        image_paths = []
        self.intrinsics_all = []
        self.pose_all = []
        for idx in self.list:
            print(idx)
            image_paths.append(image_path[idx])
            self.intrinsics_all.append(self.intrinsics_alls[idx])
            self.pose_all.append(self.pose_alls[idx])

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

        self.n_images = len(self.rgb_images)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # [C, H, W], color range [0, 255] -> [0, 1]
                transforms.Resize([256, 256]),
                transforms.Normalize([0.5], [0.5]),  # color range [0, 1] -> [-1, 1]
            ]
        )

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

