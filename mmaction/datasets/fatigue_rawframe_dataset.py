import copy
import os.path as osp

import torch

from .base import BaseDataset
from .builder import DATASETS

import numpy as np
import json
import os
import random
from pathlib import Path

@DATASETS.register_module()
class FatigueRawframeDataset(BaseDataset):
    """Rawframe dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the label of a video, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        some/directory-1 163 1
        some/directory-2 122 1
        some/directory-3 258 2
        some/directory-4 234 2
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a multi-class annotation file:


    .. code-block:: txt

        some/directory-1 163 1 3 5
        some/directory-2 122 1 2
        some/directory-3 258 2
        some/directory-4 234 2 4 6 8
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a with_offset annotation file (clips from long videos), each
    line indicates the directory to frames of a video, the index of the start
    frame, total frames of the video clip and the label of a video clip, which
    are split with a whitespace.


    .. code-block:: txt

        some/directory-1 12 163 3
        some/directory-2 213 122 4
        some/directory-3 100 258 5
        some/directory-4 98 234 2
        some/directory-5 0 295 3
        some/directory-6 50 121 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        with_offset (bool): Determines whether the offset information is in
            ann_file. Default: False.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Default: False.
        num_classes (int | None): Number of classes in the dataset.
            Default: None.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
            Default: 'RGB'.
        sample_by_class (bool): Sampling by class, should be set `True` when
            performing inter-class data balancing. Only compatible with
            `multi_class == False`. Only applies for training. Default: False.
        power (float): We support sampling data with the probability
            proportional to the power of its label frequency (freq ^ power)
            when sampling data. `power == 1` indicates uniformly sampling all
            data; `power == 0` indicates uniformly sampling all classes.
            Default: 0.
        dynamic_length (bool): If the dataset length is dynamic (used by
            ClassSpecificDistributedSampler). Default: False.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 test_all=False,
                 test_save_label_path=None,
                 test_save_results_path=None,
                 filename_tmpl='img_{:05}.jpg',
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0.,
                 dynamic_length=False,
                 min_frames_before_fatigue=32):
        self.filename_tmpl = filename_tmpl
        self.with_offset = with_offset
        self.min_frames_before_fatigue = min_frames_before_fatigue
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            test_all,
            test_save_label_path,
            test_save_results_path,
            multi_class,
            num_classes,
            start_index,
            modality,
            sample_by_class=sample_by_class,
            power=power,
            dynamic_length=dynamic_length)

    def get_valid_fatigue_idx(self, rect_infos, min_frames_before_fatigue, fatigue_idxs_str):
        global no_facerect_count
        global no_file_list

        # prepare fatigue index
        fatigue_idxs = [int(x) for x in fatigue_idxs_str]

        # prepare face rectangle idx map info

        idx_rect_map = np.zeros(len(rect_infos), np.bool)
        for info in rect_infos:
            idx = int(info.split('.')[0].split('_')[1]) - 1
            if not rect_infos[info] is None:
                idx_rect_map[idx] = True

        # index check
        min_frames_before_fatigue = min_frames_before_fatigue
        valid_idxs = []
        for fat_end_idx in fatigue_idxs:
            fat_end_idx -= 1

            # # just for test, remember to del !!!!
            # fat_end_idx += 10

            fat_start_idx = max(fat_end_idx - min_frames_before_fatigue + 1, 0)

            if idx_rect_map[fat_start_idx:fat_end_idx + 1].sum() == min_frames_before_fatigue:
                # valid idx
                valid_idxs.append(fat_end_idx + 1)

        return valid_idxs

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        video_infos = []
        flatten_video_infos = []
        print("Start to Parsing label file ", self.ann_file)

        # statistics info
        statistics_info = {}
        statistics_info['total'] = 0
        statistics_info['invalid'] = 0

        statistics_info[0] = {}
        statistics_info[1] = {}
        statistics_info[0]['num'] = 0
        statistics_info[0]['invalid'] = 0
        statistics_info[0]['clips'] = 0
        statistics_info[1]['num'] = 0
        statistics_info[1]['invalid'] = 0
        statistics_info[1]['clips'] = 0

        with open(self.ann_file, 'r') as fin:
            for line in fin:
                # video_prefix, total_frame_num, label, fatigue indexes
                tmp_split = line.strip().split(',')
                line_split = tmp_split[:3]
                fatigue_idxs_str = tmp_split[3:]

                video_prefix = line_split[0]
                video_prefix = video_prefix.replace('\\', '/')
                total_num = int(line_split[1])  # have no use
                fat_label = int(line_split[2])
                video_path = os.path.join(self.data_prefix, video_prefix)

                # statistics
                statistics_info['total'] += 1
                statistics_info[fat_label]['num'] += 1

                #video_path = Path(self.data_prefix, video_prefix)
                # get face_rectangle_infos
                facerect_file_path = os.path.join(video_path, 'facerect.npy')
                if not os.path.exists(facerect_file_path):
                    statistics_info['invalid'] += 1
                    statistics_info[fat_label]['invalid'] += 1
                    print("!! Can not found ", facerect_file_path)
                    continue
                rect_infos = np.load(facerect_file_path, allow_pickle=True).item()
                fat_idxs = self.get_valid_fatigue_idx(rect_infos, self.min_frames_before_fatigue, fatigue_idxs_str)
                if len(fat_idxs)<1:
                    statistics_info['invalid'] += 1
                    statistics_info[fat_label]['invalid'] += 1
                    continue

                # get each fatigue to video info
                infos = []
                for fat_end_idx in fat_idxs:
                    video_info = {}
                    video_info['facerect_infos'] = rect_infos
                    video_info['fat_idxs'] = fat_idxs
                    # idx for frame_dir
                    frame_dir = video_prefix
                    if self.data_prefix is not None:
                        frame_dir = osp.join(self.data_prefix, frame_dir)
                    video_info['frame_dir'] = frame_dir
                    # get start_index
                    video_info['start_index'] = fat_end_idx - self.min_frames_before_fatigue + 1
                    video_info['total_frames'] = self.min_frames_before_fatigue

                    # idx for label[s]
                    label = [fat_label]
                    assert label, f'missing label in line: {line}'
                    if self.multi_class:
                        assert self.num_classes is not None
                        video_info['label'] = label
                    else:
                        assert len(label) == 1
                        video_info['label'] = label[0]
                    infos.append(video_info)
                video_infos.append(infos)
                flatten_video_infos.extend(infos)
                # statistics
                statistics_info[fat_label]['clips'] += len(fat_idxs)

        print("Total {}\nInvalid {}\n\nFatigue_close {}\nInvalid {}\nValid {}\nClips {}, Clips_per_Video {}\n\nFatigue_look_down {}\nInvalid {}\nValid {}\nClips {}, Clips_per_Video {}\nflatten_video_infos {}".format(
            statistics_info['total'], statistics_info['invalid'],
            statistics_info[1]['num'], statistics_info[1]['invalid'], statistics_info[1]['num']-statistics_info[1]['invalid'], statistics_info[1]['clips'], statistics_info[1]['clips']/max(1, statistics_info[1]['num']-statistics_info[1]['invalid']),
            statistics_info[0]['num'], statistics_info[0]['invalid'], statistics_info[0]['num'] - statistics_info[0]['invalid'], statistics_info[0]['clips'], statistics_info[0]['clips']/max(1, statistics_info[0]['num']-statistics_info[0]['invalid']),
            len(flatten_video_infos)
        ))

        return video_infos,flatten_video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(random.choice(self.video_infos[idx]))
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        #results['start_index'] = self.start_index

        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        if self.test_all:
            results = copy.deepcopy(self.video_infos[idx])
        else:
            results = copy.deepcopy(self.video_infos[idx][0])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        #results['start_index'] = self.start_index

        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

@DATASETS.register_module()
class FatigueCleanDataset(BaseDataset):
    """Rawframe dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the label of a video, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        some/directory-1 163 1
        some/directory-2 122 1
        some/directory-3 258 2
        some/directory-4 234 2
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a multi-class annotation file:


    .. code-block:: txt

        some/directory-1 163 1 3 5
        some/directory-2 122 1 2
        some/directory-3 258 2
        some/directory-4 234 2 4 6 8
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a with_offset annotation file (clips from long videos), each
    line indicates the directory to frames of a video, the index of the start
    frame, total frames of the video clip and the label of a video clip, which
    are split with a whitespace.


    .. code-block:: txt

        some/directory-1 12 163 3
        some/directory-2 213 122 4
        some/directory-3 100 258 5
        some/directory-4 98 234 2
        some/directory-5 0 295 3
        some/directory-6 50 121 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        with_offset (bool): Determines whether the offset information is in
            ann_file. Default: False.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Default: False.
        num_classes (int | None): Number of classes in the dataset.
            Default: None.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
            Default: 'RGB'.
        sample_by_class (bool): Sampling by class, should be set `True` when
            performing inter-class data balancing. Only compatible with
            `multi_class == False`. Only applies for training. Default: False.
        power (float): We support sampling data with the probability
            proportional to the power of its label frequency (freq ^ power)
            when sampling data. `power == 1` indicates uniformly sampling all
            data; `power == 0` indicates uniformly sampling all classes.
            Default: 0.
        dynamic_length (bool): If the dataset length is dynamic (used by
            ClassSpecificDistributedSampler). Default: False.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 video_data_prefix,
                 facerect_data_prefix,
                 data_phase='train',
                 test_mode=False,
                 test_all=False,
                 test_save_label_path=None,
                 test_save_results_path=None,
                 filename_tmpl='img_{:05}.jpg',
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0.,
                 dynamic_length=False,
                 min_frames_before_fatigue=32):
        self.filename_tmpl = filename_tmpl
        self.with_offset = with_offset
        self.min_frames_before_fatigue = min_frames_before_fatigue
        self.video_data_prefix = video_data_prefix
        self.facerect_data_prefix = facerect_data_prefix
        self.data_phase = data_phase
        super().__init__(
            ann_file,
            pipeline,
            video_data_prefix,
            test_mode,
            test_all,
            test_save_label_path,
            test_save_results_path,
            multi_class,
            num_classes,
            start_index,
            modality,
            sample_by_class=sample_by_class,
            power=power,
            dynamic_length=dynamic_length)

    def get_valid_fatigue_idx(self, rect_infos, min_frames_before_fatigue, fatigue_idxs, video_dir, max_frames=500):
        global no_facerect_count
        global no_file_list

        # prepare face rectangle idx map info
        idx_rect_map = np.zeros(max_frames+1, np.bool)
        for info in rect_infos:
            # just ignore index 0, both images and fatigue index are start with index 1
            idx = int(info.split('.')[0].split('_')[1])
            if not rect_infos[info] is None and os.path.exists(os.path.join(video_dir, info)):
                idx_rect_map[idx] = True

        # index check
        min_frames_before_fatigue = min_frames_before_fatigue
        valid_idxs = []
        for fat_end_idx in fatigue_idxs:
            fat_start_idx = max(fat_end_idx - min_frames_before_fatigue + 1, 1)
            # border control
            if idx_rect_map[fat_start_idx:fat_end_idx + 1].sum() == min_frames_before_fatigue:
                # valid idx
                valid_idxs.append(fat_end_idx)

        return valid_idxs

    def get_facerects(self, filepath):
        facerect_infos = {}
        with open(filepath, 'r') as fp:
            anns = json.load(fp)
            for imgname in anns:
                facerect_infos[imgname] = anns[imgname]['bbox']
        return facerect_infos

    def load_annotations(self):
        video_infos = []
        flatten_video_infos = []
        label_map = {'fatigue_close': 1, 'fatigue_look_down': 0}
        print("Start to Parsing label file ", self.ann_file)

        # statistics info
        statistics_info = {}
        statistics_info['total'] = 0
        statistics_info['invalid'] = 0

        statistics_info[0] = {}
        statistics_info[1] = {}
        statistics_info[0]['num'] = 0
        statistics_info[0]['invalid'] = 0
        statistics_info[0]['clips'] = 0
        statistics_info[1]['num'] = 0
        statistics_info[1]['invalid'] = 0
        statistics_info[1]['clips'] = 0

        with open(self.ann_file, 'r') as fp:
            anns = json.load(fp)

            for vname in anns:
                vinfo = anns[vname]

                # check if train or valid
                if self.data_phase != vinfo['license_plate_type']:
                    continue

                # video path
                video_path = os.path.join(self.video_data_prefix, vname)

                # total frames
                total_frames = vinfo['frames_avi']

                # video label
                fat_label = label_map[vinfo['label']]

                # fatigue_idxs
                fatigue_idxs = vinfo['fatigue_warning_idx']

                # statistics
                statistics_info['total'] += 1
                statistics_info[fat_label]['num'] += 1

                # video face rectangle
                facerect_path = os.path.join(self.facerect_data_prefix, vname + '.json')
                if not os.path.exists(facerect_path):
                    statistics_info['invalid'] += 1
                    statistics_info[fat_label]['invalid'] += 1
                    print("!! Can not found ", facerect_path)
                    continue
                rect_infos = self.get_facerects(facerect_path)

                # get valid fatigue index according to facerect infos and fatigue index
                fat_idxs = self.get_valid_fatigue_idx(rect_infos, self.min_frames_before_fatigue, fatigue_idxs, video_path, max_frames=total_frames)
                if len(fat_idxs) < 1:
                    statistics_info['invalid'] += 1
                    statistics_info[fat_label]['invalid'] += 1
                    continue

                # extract all clips info in the video
                infos = []
                for fat_end_idx in fat_idxs:
                    video_info = {}
                    video_info['facerect_infos'] = rect_infos
                    video_info['fat_idxs'] = fat_idxs
                    # idx for frame_dir
                    video_info['frame_dir'] = video_path
                    # get start_index
                    video_info['start_index'] = fat_end_idx - self.min_frames_before_fatigue + 1
                    video_info['total_frames'] = self.min_frames_before_fatigue
                    # idx for label[s]
                    video_info['label'] = fat_label

                    infos.append(video_info)
                video_infos.append(infos)
                flatten_video_infos.extend(infos)
                # statistics
                statistics_info[fat_label]['clips'] += len(fat_idxs)

                # debug
                # if len(video_infos)==50:
                #     break

        print(
            "Total {}\nInvalid {}\n\nFatigue_close {}\nInvalid {}\nValid {}\nClips {}, Clips_per_Video {}\n\nFatigue_look_down {}\nInvalid {}\nValid {}\nClips {}, Clips_per_Video {}\nflatten_video_infos {}".format(
                statistics_info['total'], statistics_info['invalid'],
                statistics_info[1]['num'], statistics_info[1]['invalid'],
                statistics_info[1]['num'] - statistics_info[1]['invalid'], statistics_info[1]['clips'],
                statistics_info[1]['clips'] / max(1, statistics_info[1]['num'] - statistics_info[1]['invalid']),
                statistics_info[0]['num'], statistics_info[0]['invalid'],
                statistics_info[0]['num'] - statistics_info[0]['invalid'], statistics_info[0]['clips'],
                statistics_info[0]['clips'] / max(1, statistics_info[0]['num'] - statistics_info[0]['invalid']),
                len(flatten_video_infos)
            ))

        return video_infos, flatten_video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(random.choice(self.video_infos[idx]))
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        #results['start_index'] = self.start_index

        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        if self.test_all:
            results = copy.deepcopy(self.video_infos[idx])
        else:
            results = copy.deepcopy(self.video_infos[idx][0])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        #results['start_index'] = self.start_index

        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

@DATASETS.register_module()
class FatigueNormalDataset(BaseDataset):
    """Rawframe dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the label of a video, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        some/directory-1 163 1
        some/directory-2 122 1
        some/directory-3 258 2
        some/directory-4 234 2
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a multi-class annotation file:


    .. code-block:: txt

        some/directory-1 163 1 3 5
        some/directory-2 122 1 2
        some/directory-3 258 2
        some/directory-4 234 2 4 6 8
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a with_offset annotation file (clips from long videos), each
    line indicates the directory to frames of a video, the index of the start
    frame, total frames of the video clip and the label of a video clip, which
    are split with a whitespace.


    .. code-block:: txt

        some/directory-1 12 163 3
        some/directory-2 213 122 4
        some/directory-3 100 258 5
        some/directory-4 98 234 2
        some/directory-5 0 295 3
        some/directory-6 50 121 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        with_offset (bool): Determines whether the offset information is in
            ann_file. Default: False.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Default: False.
        num_classes (int | None): Number of classes in the dataset.
            Default: None.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
            Default: 'RGB'.
        sample_by_class (bool): Sampling by class, should be set `True` when
            performing inter-class data balancing. Only compatible with
            `multi_class == False`. Only applies for training. Default: False.
        power (float): We support sampling data with the probability
            proportional to the power of its label frequency (freq ^ power)
            when sampling data. `power == 1` indicates uniformly sampling all
            data; `power == 0` indicates uniformly sampling all classes.
            Default: 0.
        dynamic_length (bool): If the dataset length is dynamic (used by
            ClassSpecificDistributedSampler). Default: False.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 video_data_prefix,
                 facerect_data_prefix,
                 data_phase='train',
                 test_mode=False,
                 test_all=False,
                 test_save_label_path=None,
                 test_save_results_path=None,
                 filename_tmpl='img_{:05}.jpg',
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0.,
                 dynamic_length=False,
                 min_frames_before_fatigue=32):
        self.filename_tmpl = filename_tmpl
        self.with_offset = with_offset
        self.min_frames_before_fatigue = min_frames_before_fatigue
        self.video_data_prefix = video_data_prefix
        self.facerect_data_prefix = facerect_data_prefix
        self.data_phase = data_phase
        super().__init__(
            ann_file,
            pipeline,
            video_data_prefix,
            test_mode,
            test_all,
            test_save_label_path,
            test_save_results_path,
            multi_class,
            num_classes,
            start_index,
            modality,
            sample_by_class=sample_by_class,
            power=power,
            dynamic_length=dynamic_length)

    def get_valid_fatigue_idx(self, rect_infos, min_frames_before_fatigue, fatigue_idxs, video_dir, max_frames=500):
        global no_facerect_count
        global no_file_list

        # prepare face rectangle idx map info
        idx_rect_map = np.zeros(max_frames+1, np.bool)
        for info in rect_infos:
            # just ignore index 0, both images and fatigue index are start with index 1
            idx = int(info.split('.')[0].split('_')[1])
            if not rect_infos[info] is None and os.path.exists(os.path.join(video_dir, info)):
                idx_rect_map[idx] = True

        # index check
        min_frames_before_fatigue = min_frames_before_fatigue
        valid_idxs = []
        for fat_end_idx in fatigue_idxs:
            fat_start_idx = max(fat_end_idx - min_frames_before_fatigue + 1, 1)
            # border control
            if idx_rect_map[fat_start_idx:fat_end_idx + 1].sum() == min_frames_before_fatigue:
                # valid idx
                valid_idxs.append(fat_end_idx)

        return valid_idxs

    def get_facerects(self, filepath):
        facerect_infos = {}
        with open(filepath, 'r') as fp:
            anns = json.load(fp)
            for imgname in anns:
                facerect_infos[imgname] = anns[imgname]['bbox']
        return facerect_infos

    def get_normal_index(self, fatigue_idxs, total_frames, min_frames_before_fatigue):
        return np.arange(min_frames_before_fatigue-1,total_frames)

    def load_annotations(self):
        video_infos = []
        flatten_video_infos = []
        label_map = {'fatigue_close': 1, 'fatigue_look_down': 0}
        print("Start to Parsing label file ", self.ann_file)

        # statistics info
        statistics_info = {}
        statistics_info['total'] = 0
        statistics_info['invalid'] = 0

        statistics_info[0] = {}
        statistics_info[1] = {}
        statistics_info[0]['num'] = 0
        statistics_info[0]['invalid'] = 0
        statistics_info[0]['clips'] = 0
        statistics_info[1]['num'] = 0
        statistics_info[1]['invalid'] = 0
        statistics_info[1]['clips'] = 0

        with open(self.ann_file, 'r') as fp:
            anns = json.load(fp)

            for vname in anns:
                vinfo = anns[vname]

                # check if train or valid
                if self.data_phase != vinfo['license_plate_type']:
                    continue

                # video path
                video_path = os.path.join(self.video_data_prefix, vname)

                # total frames
                total_frames = vinfo['frames_avi']

                # video label
                fat_label = label_map[vinfo['label']]

                # fatigue_idxs
                fatigue_idxs = vinfo['fatigue_warning_idx']

                # NEW: add normal training sample from fatigue lookdown videos
                if fat_label == 0:
                    fatigue_idxs = self.get_normal_index(fatigue_idxs, total_frames, self.min_frames_before_fatigue)

                # statistics
                statistics_info['total'] += 1
                statistics_info[fat_label]['num'] += 1

                # video face rectangle
                facerect_path = os.path.join(self.facerect_data_prefix, vname + '.json')
                if not os.path.exists(facerect_path):
                    statistics_info['invalid'] += 1
                    statistics_info[fat_label]['invalid'] += 1
                    print("!! Can not found ", facerect_path)
                    continue
                rect_infos = self.get_facerects(facerect_path)

                # get valid fatigue index according to facerect infos and fatigue index
                fat_idxs = self.get_valid_fatigue_idx(rect_infos, self.min_frames_before_fatigue, fatigue_idxs, video_path, max_frames=total_frames)
                if len(fat_idxs) < 1:
                    statistics_info['invalid'] += 1
                    statistics_info[fat_label]['invalid'] += 1
                    continue

                # extract all clips info in the video
                infos = []
                for fat_end_idx in fat_idxs:
                    video_info = {}
                    video_info['facerect_infos'] = rect_infos
                    video_info['fat_idxs'] = fat_idxs
                    # idx for frame_dir
                    video_info['frame_dir'] = video_path
                    # get start_index
                    video_info['start_index'] = fat_end_idx - self.min_frames_before_fatigue + 1
                    video_info['total_frames'] = self.min_frames_before_fatigue
                    # idx for label[s]
                    video_info['label'] = fat_label

                    infos.append(video_info)
                video_infos.append(infos)
                flatten_video_infos.extend(infos)
                # statistics
                statistics_info[fat_label]['clips'] += len(fat_idxs)

                # debug
                # if len(video_infos)==50:
                #     break

        print(
            "Total {}\nInvalid {}\n\nFatigue_close {}\nInvalid {}\nValid {}\nClips {}, Clips_per_Video {}\n\nFatigue_look_down {}\nInvalid {}\nValid {}\nClips {}, Clips_per_Video {}\nflatten_video_infos {}".format(
                statistics_info['total'], statistics_info['invalid'],
                statistics_info[1]['num'], statistics_info[1]['invalid'],
                statistics_info[1]['num'] - statistics_info[1]['invalid'], statistics_info[1]['clips'],
                statistics_info[1]['clips'] / max(1, statistics_info[1]['num'] - statistics_info[1]['invalid']),
                statistics_info[0]['num'], statistics_info[0]['invalid'],
                statistics_info[0]['num'] - statistics_info[0]['invalid'], statistics_info[0]['clips'],
                statistics_info[0]['clips'] / max(1, statistics_info[0]['num'] - statistics_info[0]['invalid']),
                len(flatten_video_infos)
            ))

        return video_infos, flatten_video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(random.choice(self.video_infos[idx]))
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        #results['start_index'] = self.start_index

        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        if self.test_all:
            results = copy.deepcopy(self.video_infos[idx])
        else:
            results = copy.deepcopy(self.video_infos[idx][0])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        #results['start_index'] = self.start_index

        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)