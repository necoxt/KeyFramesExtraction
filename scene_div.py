# -*- coding: utf-8 -*-
import argparse

import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import argrelextrema
from glob import glob
import os
from tqdm import tqdm

# import torch
# from torch.utils.data.dataloader import default_collate
# from torch.utils.data import Dataset, DataLoader


def smooth(x, window_len=13, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """
    print(f"Number of frames: {len(x)}, Window Length: {window_len}")
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]


def rel_change(a, b):
    x = (b - a) / max(a, b)
    print(x)
    return x


class FileLoader:
    def __init__(self, path, type='jpg'):
        assert type in ["mp4", "jpg", "png"], f"Supported types: mp4, jpg, png. Found {type}."
        self.type = type
        self.data = sorted(glob(os.path.join(path, "**/*." + type), recursive=True))
        if len(self.data) == 0:
            raise FileNotFoundError(f"Unable to find .{type} file in {path}")
        self.loader = self._loader()

    def __iter__(self):
        return self

    def __next__(self):
        return self.loader.__next__()

    def __len__(self):
        return len(self.data)

    def _loader(self):
        if self.type == "mp4":
            for path in self.data:
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
                i = 0
                while ret:
                    luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
                    yield luv, path + ' ' + str(i)
                    ret, frame = cap.read()
                    i += 1
                cap.release()
        elif self.type in ["jpg", "png"]:
            for i, path in enumerate(self.data):
                img = cv2.imread(path)
                if img is None:
                    continue
                luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
                yield luv, self.data[i]
        else:
            raise NotImplementedError

# def collate_fn(batch):
#     # Filter None data
#     if batch[0][0] is None:
#         return None, None
#     data, path = default_collate(batch)
#     return data.squeeze().numpy(), path
#
#
# class FileDataset(Dataset):
#     def __init__(self, path, type='jpg'):
#         assert type in ["jpg", "png"], f"Supported types: jpg, png. Found {type}."
#         self.type = type
#         self.data = sorted(glob(os.path.join(path, "**/*." + type), recursive=True))
#         if len(self.data) == 0:
#             raise FileNotFoundError(f"Unable to find .{type} file in {path}")
#
#     def __getitem__(self, item):
#         img = cv2.imread(self.data[item])
#         if img is None:
#             return None, None
#         luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
#         return luv, self.data[item]
#
#     def __len__(self):
#         return len(self.data)


def write_txt(out_dir, key_frames, split):
    key_frames = [i + ' ' + i.replace('color', 'depth').replace('jpg', 'png') for i in key_frames]
    train_file = os.path.join(out_dir, "azure_train.txt")
    test_file = os.path.join(out_dir, "azure_test.txt")
    l_test = int(len(key_frames) * split)
    l_train = len(key_frames) - l_test
    with open(train_file, 'w') as f:
        f.writelines([kf + "\n" for kf in key_frames[:l_train]])
        print(f"Extracted Training Keyframes: {l_train}")
    with open(test_file, 'w') as f:
        f.writelines([kf + "\n" for kf in key_frames[:l_test]])
        print(f"Extracted Testing Keyframes: {l_test}")


def extract_frames(path, out_dir, len_window, split, view, type='jpg', mode="USE_LOCAL_MAXIMA", value=None):
    assert mode in ["USE_TOP_ORDER", "USE_THRESH", "USE_LOCAL_MAXIMA"]
    if mode in ["USE_TOP_ORDER", "USE_THRESH"]:
        assert value is not None
    print("Data path:" + path)
    print("Output path: " + out_dir)

    file_loader = FileLoader(path, type)
    # file_data = FileDataset(path, type)
    # file_loader = DataLoader(file_data, batch_size=1, shuffle=False, num_workers=8, collate_fn=collate_fn)
    curr_frame = None
    prev_frame = None

    frames = []

    # Get absdiff between frames
    for i, (curr_frame, path) in enumerate(tqdm(file_loader)):
        if curr_frame is not None and prev_frame is not None:
            # logic here
            count = cv2.absdiff(curr_frame, prev_frame).sum()
            # frame = Frame(i+1, path, count)
            frames.append((count, path))
        prev_frame = curr_frame

    # Extract keyframe
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Unable to find folder {out_dir}")
    if mode == "USE_TOP_ORDER":
        # sort the list in descending order
        frames.sort(reverse=True)
        key_frames = [frame for count, frame in frames[:value]]
        write_txt(out_dir, key_frames, split)
    elif mode == "USE_THRESH":
        print("Using Threshold")
        key_frames = [frames[i][1] for i in range(1, len(frames)) if
                      rel_change(np.float(frames[i - 1][0]), np.float(frames[i][0])) >= value]
        write_txt(out_dir, key_frames, split)
    elif mode == "USE_LOCAL_MAXIMA":
        print("Using Local Maxima")
        diff_array = np.array([count for count, _ in frames])
        sm_diff_array = smooth(diff_array, len_window)
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
        key_frames = [frames[i][1] for i in frame_indexes]
        key_frames_counts = [frames[i][0] for i in frame_indexes]
        write_txt(out_dir, key_frames, split)

        plt.figure(figsize=(40, 20))
        plt.locator_params(numticks=100)
        plt.stem(frame_indexes, key_frames_counts, linefmt='red')
        plt.plot(sm_diff_array)
        plt.show()

        # for i, path in enumerate(key_frames):
        #     cv2.imshow(path, cv2.imread(path))
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True, help='video or image folder data path')
    parser.add_argument('-o', '--out', default="./", help='output directory of extracted frames')
    parser.add_argument('-w', '--window', default=13, type=int, help='smoothing window size')
    parser.add_argument('-s', '--split', default=0.2, type=float, help='ratio of test set')
    parser.add_argument('-v', '--view', action="store_true", help='view results')
    args = parser.parse_args(['--path', './AzureKinect/20220506103628', '--window', '5'])
    extract_frames(args.path, args.out, args.window, args.split, args.view)
