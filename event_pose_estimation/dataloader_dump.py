import os
import sys
import numpy as np
import cv2
from torch.utils.data import Dataset
import pickle
import joblib
import torch
sys.path.append("..")
from flow_net.flowlib import flow_to_image


class TrackingDataloader(Dataset):
    def __init__(
            self,
            data_dir='data_event',
            max_steps=16,
            num_steps=8,
            skip=2,
            events_input_channel=8,
            img_size=256,
            mode='train',
            use_flow=True,
            use_flow_rgb=False,
            use_hmr_feats=False,
            use_vibe_init=False,
            use_hmr_init=False
    ):
        self.data_dir = data_dir
        self.events_input_channel = events_input_channel
        self.skip = skip
        self.max_steps = max_steps
        self.num_steps = num_steps
        self.img_size = img_size
        scale = self.img_size / 1280.
        self.cam_intr = np.array([1679.3, 1679.3, 641, 641]) * scale
        self.use_hmr_feats = use_hmr_feats
        self.use_flow = use_flow
        self.use_flow_rgb = use_flow_rgb
        self.use_vibe_init = use_vibe_init
        self.use_hmr_init = use_hmr_init

        self.mode = mode
        if os.path.exists('%s/%s_track%02i%02i.pkl' % (self.data_dir, self.mode, self.num_steps, self.skip)):
            self.all_clips = pickle.load(
                open('%s/%s_track%02i%02i.pkl' % (self.data_dir, self.mode, self.num_steps, self.skip), 'rb'))
        else:
            self.all_clips = self.obtain_all_clips()

        if self.use_vibe_init:
            print('[VIBE init]')
            all_clips = []
            for (action, frame_idx) in self.all_clips:
                if os.path.exists('%s/vibe_results_%02i%02i/%s/fullpic%04i_vibe%02i.pkl' %
                                  (self.data_dir, self.num_steps, self.skip, action, frame_idx, self.num_steps)):
                    all_clips.append((action, frame_idx))
                else:
                    print('[vibe not exist] %s %i' % (action, frame_idx))
            self.all_clips = all_clips

        if self.use_hmr_init:
            print('[hmr init]')
            all_clips = []
            for (action, frame_idx) in self.all_clips:
                if os.path.exists('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (self.data_dir, action, frame_idx)):
                    all_clips.append((action, frame_idx))
                else:
                    print('[hmr not exist] %s %i' % (action, frame_idx))
            self.all_clips = all_clips

        print('[%s] %i clips, track%02i%02i.pkl' %
              (self.mode, len(self.all_clips), self.num_steps, self.skip))

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, idx):
        action, frame_idx = self.all_clips[idx]
        beta, theta, tran, joints3d, joints2d = joblib.load('%s/pose_events/%s/pose%04i.pkl' % (self.data_dir, action, frame_idx))

        import itertools
        annot = {
            'images': {
                'file_name': 'full_pic_%i/%s/fullpic%04i.jpg' % (self.img_size, action, frame_idx),
                'event_frame_name': 'events_%i/%s/event%04i.png' % (self.img_size, action, frame_idx),
                'height': self.img_size,
                'width': self.img_size,
                'id': idx,
            },
            'annotations': {
                'segmentation': [],
                'keypoints': list(itertools.chain(*joints2d.tolist())),
                'joints3d': list(itertools.chain(*joints3d.tolist())),
                'beta': list(itertools.chain(*beta.tolist())),
                'theta': list(itertools.chain(*theta.tolist())),
                'tran': list(itertools.chain(*tran.tolist())),
                'num_keypoints': 24,
                'area': 0,
                'iscrowd': 0,
                'image_id': idx,
                'bbox': [0, 0, self.img_size, self.img_size],
                'category_id': 1,
                'id': idx,
            },
        }
        # return one_sample
        return annot

    def obtain_all_clips(self):
        all_clips = []
        sorted_folder_names = sorted(
            os.listdir('%s/pose_events' % self.data_dir))
        '''
        data_event/data_event_out/pose_events/
        |-- subject01_group1_time1
        |-- subject01_group1_time2
        |-- subject01_group1_time3
        '''
        action_names = []
        for action in sorted_folder_names:
            subject = action.split('_')[0]
            # split the dataset into train and test, by subject
            if self.mode == 'test':  # only test on subject 1, 2, 7
                if subject in ['subject01', 'subject02', 'subject07']:
                    action_names.append(action)
            else:  # train on datasets other than subject 1, 2, 7
                if subject not in ['subject01', 'subject02', 'subject07']:
                    action_names.append(action)

        for action in action_names:
            if not os.path.exists('%s/pose_events/%s/pose_info.pkl' % (self.data_dir, action)):
                print('[warning] not exsit %s/pose_events/%s/pose_info.pkl' %
                      (self.data_dir, action))
                continue
            frame_indices = joblib.load(
                '%s/pose_events/%s/pose_info.pkl' % (self.data_dir, action))
            for i in range(len(frame_indices) - self.max_steps * self.skip):
                frame_idx = frame_indices[i]
                end_frame_idx = frame_idx + self.max_steps * self.skip
                if not os.path.exists('%s/pred_flow_events_%i/%s/flow%04i.pkl' % (self.data_dir, self.img_size, action, end_frame_idx)):
                    # print('flow %i not exists for %s-%i' % (end_frame_idx, action, frame_idx))
                    continue
                if not os.path.exists('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (self.data_dir, action, frame_idx)):
                    continue
                if end_frame_idx == frame_indices[i + self.max_steps * self.skip]:
                    # action, frame_idx
                    all_clips.append((action, frame_idx))

        pickle.dump(
            all_clips,
            open('%s/%s_track%02i%02i.pkl' %
                 (self.data_dir, self.mode, self.num_steps, self.skip), 'wb')
        )
        return all_clips



if __name__ == '__main__':

    # accept an argument to specify the mode
    specified_mode = sys.argv[1] # 'train' or 'test'

    os.environ['OMP_NUM_THREADS'] = '1'
    data_loader = TrackingDataloader(
        data_dir='/root/EventHPE/data_event/data_event_out',
        max_steps=16,
        num_steps=8,
        skip=2,
        events_input_channel=8,
        img_size=256,
        mode=specified_mode,
        use_hmr_feats=True
    )

    keypoinsts_annotations = {
        'images': [],
        'annotations': [],
        'categories': [{
            'supercategory': 'person',
            'id': 1,
            'name': 'person',
        }],
    }
    
    from tqdm import tqdm
    for i, sample in tqdm(enumerate(data_loader), total=len(data_loader)):
        keypoinsts_annotations['images'].append(sample['images'])
        keypoinsts_annotations['annotations'].append(sample['annotations'])

    print('total number of images: ', len(keypoinsts_annotations['images']))
    # dump annotations as json file

    import json
    file_path = f'/root/EventHPE/data_event/data_event_out/annotations/keypoints_annotations_{specified_mode}.json'

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    json.dump(keypoinsts_annotations, open(file_path, 'w'))

    print('done')

    # sample = data_train[10000]
    # data_train.visualize(20000)
    # print()
    # for k, v in sample.items():
    #     if k is not 'info':
    #         print(k, v.size())

    # data_test = TrackingDataloader(
    #     data_dir='/home/shihao/data_event',
    #     max_steps=16,
    #     num_steps=8,
    #     skip=2,
    #     events_input_channel=8,
    #     img_size=256,
    #     mode='train',
    #     use_flow=True,
    #     use_flow_rgb=False,
    #     use_hmr_feats=False,
    #     use_vibe_init=False,
    #     use_hmr_init=True,
    # )
    # # data_test.visualize(30000)
    # sample = data_test[30000]
    # for k, v in sample.items():
    #     if k != 'info':
    #         print(k, v.size())
