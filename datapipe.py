import os 
import numpy as np 
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import scipy.io as sio
import glob  
    
subjects = 15 # Num. of subjects used for LOSO
classes = 3 # Num. of classes 

def to_categorical(y, num_classes=None, dtype='float32'): 
    #one-hot encoding
    y = np.array(y, dtype='int16')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0] 
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class EmotionDataset(InMemoryDataset):
    def __init__(self, stage, root, subjects, sub_i, X=None, Y=None, edge_index=None,
                 transform=None, pre_transform=None,ADJ=None):
        self.stage = stage #Train or test
        self.subjects = subjects  
        self.sub_i = sub_i
        self.X = X
        self.Y = Y
        self.edge_index = edge_index
        self.ADJ=ADJ
        
        super(EmotionDataset, self).__init__(root, transform, pre_transform)
        #super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(
                1, self.stage, self.subjects, self.sub_i)]
    def download(self):
        pass
    
    def process(self): 
        data_list = [] 
        # process by samples
        num_samples = np.shape(self.Y)[0]
        for sample_id in tqdm(range(num_samples)): 
            x = self.X[sample_id,:]    
            x = torch.FloatTensor(x)
            adj=self.ADJ[sample_id,:]
            adj=torch.FloatTensor(adj)
            y = torch.FloatTensor(self.Y[sample_id,:])
            data = Data(x=x, y=y,adj=adj)

            data_list.append(data) 
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
def normalize(data):
    mee=np.mean(data,0)
    data=data-mee
    stdd=np.std(data,0)
    data=data/(stdd+1e-7)
    return data  
def get_data():
    path0 = 'algdatas/'
    path='Myldsdbde/'
    path00='Mydbspp/'
    label = sio.loadmat(path0+'label.mat')['label']
    print(label)
    files = sorted(glob.glob(path+'*_*'))

    sublist = set()
    for f in files:
        sublist.add(f.split('/')[-1].split('_')[0] )
    
    print('Total number of subjects: {:.0f}'.format(len(sublist)))
 
   
    sub_de = [] 
    sub_adj=[]
    sub_label = []
     
    for sub_i in range(subjects):
        sub = str(sub_i+1)
        sub_files = glob.glob(path+sub+'_*')
        sub_files00=glob.glob(path00+sub+'_*')
        mov_de = []
        mov_adj=[]
        mov_label=[] 
                
        sub_files=sorted(sub_files)
        sub_files00=sorted(sub_files00)
        for f in range(0,1):
            deee=sub_files[f]
            adjj=sub_files00[f]
            print(deee)
            print(adjj)
            datade = sio.loadmat(deee, verify_compressed_data_integrity=False)
            dataadj= sio.loadmat(adjj,verify_compressed_data_integrity=False)
            dekeys = datade.keys()
            adjkeys= dataadj.keys()
            de_mov = [k for k in dekeys if 'dee_' in k] 
            adj_mov= [k for k in adjkeys if 'spp_' in k]   
            bb=[0,7,8,9,10,11,12,13,14,1,2,3,4,5,6]
           
            mov_dei = [] 
            mov_adjj=[]
            mov_labelk=[] 
           
            print(de_mov)
            print(adj_mov)
            for t in range(15):
                print(de_mov[bb[t]]+'----'+adj_mov[bb[t]]) 
       #         temp_de = datade[de_mov[t]].transpose(1,2,0)
                temp_de = datade[de_mov[bb[t]]].transpose(2,0,1)
                temp_adj= dataadj[adj_mov[bb[t]]].transpose(3,0,1,2)
                data_length  = temp_de.shape[0]
              
                mov_i=temp_de.reshape(data_length,5*62)
                mov_j=temp_adj.reshape(data_length,5*62*62)
                mov_k=np.full((data_length,1),label[0][t])

                mov_dei.append(mov_i)
                mov_adjj.append(mov_j)           
                mov_labelk.append(mov_k)
                       
            mov_dei=np.vstack(mov_dei).reshape(-1,5*62)
            mov_adjj=np.vstack(mov_adjj).reshape(-1,5*62*62)
            mov_labelk=np.vstack(mov_labelk).reshape(-1,1)
            

            mov_de.append(mov_dei) 
            mov_adj.append(mov_adjj)
            mov_label.append(mov_labelk)
 
        mov_de = np.vstack(mov_de) 
#        mov_de = normalize(mov_de) 
        mov_adj=np.vstack(mov_adj)
        mov_label=np.vstack(mov_label)
  
        sub_de.append(mov_de)
        sub_adj.append(mov_adj)
        sub_label.append(mov_label)
       
        
    sub_de = np.array(sub_de) 
    sub_adj=np.array(sub_adj)
    sub_label = np.array(sub_label)
   

    return sub_de,sub_adj, sub_label
      
    
def build_dataset(subjects):
    load_flag = True
    for sub_i in range(subjects):
        path = 'processed/V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(
                1, 'Train', subjects, sub_i)
        print(path)
        
        if not os.path.exists(path): 
        
            if load_flag:
                mov_coefs, adjs, labels = get_data()
                used_coefs = mov_coefs
                load_flag = False
            
            index_list = list(range(subjects))
            del index_list[sub_i]
            test_index = sub_i
            train_index = index_list
            
            print('Building train and test dataset')
            #get train & test
           
            print(used_coefs.shape)
            print(train_index) 
    #        X = used_coefs[train_index,:].reshape(-1,1, 5*62)
    #        A= adjs[train_index,:].reshape(-1,1,5*62*62)
    #        Y = labels[train_index,:].reshape(-1)
            testX = used_coefs[test_index,:].reshape(-1,1, 5*62)
            testA=adjs[test_index,:].reshape(-1,1,5*62*62)
            testY = labels[test_index,:].reshape(-1)
            #get labels
     #       _, Y = np.unique(Y, return_inverse=True)
    #        Y = to_categorical(Y, classes)
            _, testY = np.unique(testY, return_inverse=True)
            testY = to_categorical(testY, classes)
  #          train_dataset = EmotionDataset('Train', 'yzn/Newworld/ger/bayern1/', subjects, sub_i, X, Y,ADJ=A)
            test_dataset = EmotionDataset('Test', './', subjects, sub_i, testX, testY,ADJ=testA)
            print('Dataset is built.')
         
           
def get_dataset(subjects, sub_i):
    path = 'processed/V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(
            1, 'Test', subjects, sub_i)
    print(path)
    if not os.path.exists(path): 
        raise IOError('Train dataset is not exist!')
    
#    train_dataset = EmotionDataset('Train', 'yzn/Newworld/ger/bayern1/', subjects, sub_i)
    test_dataset = EmotionDataset('Test', './', subjects, sub_i)

    return 0, test_dataset



#build_dataset(15)
