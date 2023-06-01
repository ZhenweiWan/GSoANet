import os, pdb
'''
d = './val_256'
os.listdir(d)
for i in os.listdir(d) :
    if len(os.listdir(os.path.join(d, i))) != 0 :
        #pdb.set_trace()
        pass
    else :
        print(i + ' is empty')
'''

del_count = 0
path = "/media/vv/2TB/all_dataset/kinetics_frame_data/val_256"
for file in os.listdir(path):
    file = path + '/' + file
    for sub_file in os.listdir(file):
        filename = file + '/' + sub_file
        if len(os.listdir(filename)) == 0:
            print(filename)
            del_count += 1
            os.rmdir(filename)
print(del_count)