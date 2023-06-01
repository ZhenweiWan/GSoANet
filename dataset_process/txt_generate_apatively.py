import os, pdb, json

d = 'classids_gzl.json'
with open(d,'r') as f:
    js = json.load(f)

def write_attach(str_, file_w):
    str_ = str_ + '\n'
    with open(file_w, 'a+') as fw:
        fw.write(str_)


#get videolist
def video_get_write(file_dataset, file_w, js):
    cls = os.listdir(file_dataset)
    vi_count = 0
    for classi in cls:
        if classi.find('_') > 0:
            class_temp = classi.replace('_',' ')
            id_cls = js['"' + class_temp + '"']
        else :
            id_cls = js[classi]
        if not (id_cls >= 0 and id_cls <= 400) :
            pdb.set_trace()
        path_classi = os.path.join(file_dataset, classi)
        videos = os.listdir(path_classi)
        cls_count = 0
        for vi in videos :
            path_v = os.path.join(path_classi, vi)
            if os.path.isdir(path_v):
                vi_count += 1
                cls_count += 1
                str_ = os.path.join(classi, vi)  + ' ' + str(len(os.listdir(path_v))) + ' ' + str(id_cls) 
                write_attach(str_, file_w)
        print(format(cls_count) + ' videos in ' + classi)
    print('----{} videos totally---'.format(vi_count))

# file_dataset = '/media/icy/work1/frames_a_xx' #frames_val_w_mkv/'
file_dataset = '/media/vv/2TB/all_dataset/kinetics_frame_data/val_256'
file_w = 'val.txt'  #'val_gzl_w_mkv_new.txt'
video_get_write(file_dataset, file_w, js)

#write into files for each video
