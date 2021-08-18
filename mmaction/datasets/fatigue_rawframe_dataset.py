import copy
import os.path as osp

import torch

from .base import BaseDataset
from .builder import DATASETS

import numpy as np
import os

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
        fatigue_idxs = [int(x) for x in fatigue_idxs_str.strip().split(',')]

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
        print("Start to Parsing label file ", self.ann_file)

        total_video_count = 0
        invalid_video_count = 0
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                total_video_count += 1

                # video_prefix, total_frame_num, label, fatigue indexes
                tmp_split = line.strip().split(',')
                line_split = tmp_split[:3]
                fatigue_idxs_str = tmp_split[3]

                video_prefix = line_split[0]
                total_num = int(line_split[1])  # have no use
                fat_label = int(line_split[2])
                video_path = os.path.join(self.data_prefix, video_prefix)
                # get face_rectangle_infos
                facerect_file_path = os.path.join(video_path, 'facerect.npy')
                if not os.path.exists(facerect_file_path):
                    invalid_video_count += 1
                    print("Can not found ", facerect_file_path)
                    continue
                rect_infos = np.load(facerect_file_path, allow_pickle=True).item()
                fat_idxs = self.get_valid_fatigue_idx(rect_infos, self.min_frames_before_fatigue, fatigue_idxs_str)
                if len(fat_idxs)<1:
                    invalid_video_count += 1
                    continue

                # get each fatigue to video info
                for fat_end_idx in fat_idxs:
                    video_info = {}
                    video_info['facerect_infos'] = rect_infos
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
                    video_infos.append(video_info)

        print("End parsing label file, Total {} videos, and {} invalid videos under min_frames_before_fatigue {}".format(
            total_video_count, invalid_video_count, self.min_frames_before_fatigue))

        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
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
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        #results['start_index'] = self.start_index

        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)
