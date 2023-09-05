import os
import json


data_file ='/data/autsl/autsl_full.json'
dictionary = {}


# "05237": {"subset": "train", "action": [77, 1, 55]}
import pickle

with open('/data/autsl/chalearn_data_train.pkl', 'rb') as f:
    dictionary['train'] = pickle.load(f)       
with open('/data/autsl/chalearn_data_test.pkl', 'rb') as f:
    dictionary['test'] = pickle.load(f)       
with open('/data/autsl/chalearn_data_val.pkl', 'rb') as f:
    dictionary['val'] = pickle.load(f)       


name_label_dict = {}
data = {}

labels = os.listdir('/data/autsl/chalearn_processed_full/color/train')


label_count = {}
for label in labels:
    label_count[label] = 0
for i in dictionary['train']:
    name_label_dict[i['name']]=i['label'] 
    data[i['name']]= {'subset':'train','action':[labels.index(i['label']),1,label_count[i['label']]]}
    label_count[i['label']]+=1    

for i in dictionary['test']:
    name_label_dict[i['name']]=i['label'] 
    data[i['name']]= {'subset':'test','action':[labels.index(i['label']),1,label_count[i['label']]]}
    label_count[i['label']]+=1    

for i in dictionary['val']:
    name_label_dict[i['name']]=i['label'] 
    data[i['name']]= {'subset':'val','action':[labels.index(i['label']),1,label_count[i['label']]]}
    label_count[i['label']]+=1    


#65097 {'subset': 'train', 'action': [481, 1, 74]}
#{'video_file': 'WLASL2000/65225.mp4',
# 'name': '65225',
# 'seq_len': 64,
# 'label': 'book'}




print(len(data))
with open(data_file, "w") as outfile:
    json.dump(data, outfile)
