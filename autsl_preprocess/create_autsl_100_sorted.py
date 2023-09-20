# sort autsl_100 by class
import os
import shutil
data_path = '/l/users/ganzorig.batnasan/data/autsl/'
dest_path = '/l/users/ganzorig.batnasan/data/autsl/autsl_100_head_hands_merged_sorted_not_resized/color'

# for i in data:
#     print(data[i])
#     if data[i]['subset']=='test':
#         os.makedirs(os.path.join(dest_path,data[i]['subset'],data[i]['label']),exist_ok=True)
#         shutil.copy(os.path.join(data_path,i+'.mp4'),os.path.join(dest_path,data[i]['subset'],data[i]['label']))

import pickle
import json


f = open(os.path.join(data_path,'100.json'))

data = json.load(f)

f.close()

train_data = pickle.load(open(os.path.join(data_path,"chalearn_data_train.pkl"), "rb"))
test_data = pickle.load(open(os.path.join(data_path,'chalearn_data_test.pkl'),"rb"))
val_data = pickle.load(open(os.path.join(data_path,'chalearn_data_val.pkl'),"rb"))

aux_path ='chalearn_processed_full_head_hands_merged/color'

train_data_labels = {}
for i in train_data:
    train_data_labels[i['name']] = i['label']
test_data_labels = {}
for i in test_data:
    test_data_labels[i['name']] = i['label']
val_data_labels = {}
for i in val_data:
    val_data_labels[i['name']] = i['label']


for i in data:
    if data[i]['subset']=='train':
    #for i in train_data_labels:
        os.makedirs(os.path.join(dest_path,'train',train_data_labels[i]),exist_ok=True)
        shutil.copy(os.path.join(data_path,aux_path,'train',train_data_labels[i],i+'.mp4'),os.path.join(dest_path,'train',train_data_labels[i]))
    elif data[i]['subset']=='test':
    #for i in test_data_labels:
        os.makedirs(os.path.join(dest_path,'test',test_data_labels[i]),exist_ok=True)
        shutil.copy(os.path.join(data_path,aux_path,'test',test_data_labels[i],i+'.mp4'),os.path.join(dest_path,'test',test_data_labels[i]))

    else:
    #for i in val_data_labels:
        os.makedirs(os.path.join(dest_path,'val',val_data_labels[i]),exist_ok=True)
        shutil.copy(os.path.join(data_path,aux_path, 'val',val_data_labels[i],i+'.mp4'),os.path.join(dest_path,'val',val_data_labels[i]))
