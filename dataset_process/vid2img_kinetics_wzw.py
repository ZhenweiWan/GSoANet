# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from __future__ import print_function, division
import os
import sys
import subprocess
from multiprocessing import Pool
from tqdm import tqdm

n_thread = 100


def vid2jpg(file_name, class_path, dst_class_path):
    # afm change
    
    if '.mkv' not in file_name and'.mp4' not in file_name:
        print(file_name + 'is skipped')
        return

        
    name, ext = os.path.splitext(file_name)
    dst_directory_path = os.path.join(dst_class_path, name)

    video_file_path = os.path.join(class_path, file_name)
    try:
        if os.path.exists(dst_directory_path):
            if not os.path.exists(os.path.join(dst_directory_path, 'img_00001.jpg')):
                subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
                print('remove {}'.format(dst_directory_path))
                os.mkdir(dst_directory_path)
            else:
                print('*** convert has been done: {}'.format(dst_directory_path))
                return
        else:
            os.mkdir(dst_directory_path)
    except:
        print(dst_directory_path)
        return
    # afm change
    #cmd = 'ffmpeg -i \"{}\" -threads 1 -vf scale=-1:331 -q:v 0 \"{}/img_%05d.jpg\"'.format(video_file_path, dst_directory_path)
    cmd = 'ffmpeg -i \"{}\" -threads 1 -vf scale=-1:256 -q:v 0 \"{}/img_%05d.jpg\"'.format(video_file_path, dst_directory_path)
    # print(cmd)
    subprocess.call(cmd, shell=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def class_process(dir_path, dst_dir_path, class_name):
    print('*' * 20, class_name, '*'*20)
    class_path = os.path.join(dir_path, class_name)
    if not os.path.isdir(class_path):
        print('*** is not a dir {}'.format(class_path))
        return

    dst_class_path = os.path.join(dst_dir_path, class_name)
    if not os.path.exists(dst_class_path):
        os.mkdir(dst_class_path)

    vid_list = os.listdir(class_path)
    vid_list.sort()
    p = Pool(n_thread)
    from functools import partial
    worker = partial(vid2jpg, class_path=class_path, dst_class_path=dst_class_path)
    for _ in tqdm(p.imap_unordered(worker, vid_list), total=len(vid_list)):
        pass
    # p.map(worker, vid_list)
    p.close()
    p.join()

    print('\n')


if __name__ == "__main__":
    # dir_path = sys.argv[1]
    # dst_dir_path = sys.argv[2]
    # dir_path = './compress/train_256'
    # dst_dir_path = '/media/icy/zbb/K400_frames_256edge/train_256'
    dir_path = '/media/vv/50076661-58e8-4bb5-a6e1-04e6d2a043b0/awei/all_dataset/VIDEO/K400/compress/val_256'
    dst_dir_path = '/media/vv/2TB/all_dataset/kinetics_frame_data/val_256'
    
    class_list = os.listdir(dir_path)
    class_list.sort()
    for class_name in class_list:
        class_process(dir_path, dst_dir_path, class_name)

    class_name = 'test'
    class_process(dir_path, dst_dir_path, class_name)
