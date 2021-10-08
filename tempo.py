import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from torch import nn
import os
from check_tempo_curve import tempoWin
import utils
import time
import json

audio_files = '/home/rich.tsai/NAS_189/home/BeatTracking/Past/datasets/ASAP/asap-dataset-1.1/audio_files.txt'
aud_dir = '/home/rich.tsai/NAS_189/home/BeatTracking/Past/datasets/ASAP/asap-dataset-1.1/'
dataset_path = os.path.dirname(audio_files)
ann_dir = '/home/rich.tsai/NAS_189/home/BeatTracking/Past/datasets/ASAP/asap-dataset-1.1/downbeats/'


class Beatdataset(Dataset):
    def __init__(self, features, labels):
        self.feature_array = features
        self.label_array = labels
    
    def __len__(self):
        return len(self.label_array)
    
    def __getitem__(self, i):
        feature = self.feature_array[i, :, :]
        label = self.label_array[i]
        return feature, label

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        #INPUT_SIZE = 128
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        self.out = nn.Linear(64, 3)

    def forward(self, x):
        r_out, (h_c, h_h) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out

def speed_detect(labels):
    sp = np.zeros(len(labels))
    for i in range(len(labels)):
        if labels[i] >= 200:
            sp[i] = 2
        elif 101 <= labels[i] < 200:
            sp[i] = 1
        else:
            sp[i] = 0
    return sp

def main():
    cuda_num = 0#int(sys.argv[1])
    cuda_str = 'cuda:'+str(cuda_num)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    
    
    if os.path.isfile('/home/rich.tsai/NAS_189/home/BeatTracking/new/speed.npy') & os.path.isfile('/home/rich.tsai/NAS_189/home/BeatTracking/new/features.npy'):
        
        features = np.load('/home/rich.tsai/NAS_189/home/BeatTracking/new/features.npy')
        speed = np.load('/home/rich.tsai/NAS_189/home/BeatTracking/new/speed.npy')
        
    else:   
        a=[]
        label = []
        check_num = 10

        with open(audio_files) as f:

            for line in f.readlines():
    #             check_num -=1
    #             if check_num ==0:
    #                 break  
                relative = os.path.normpath(line.strip('\n'))
                audio_file = os.path.join(dataset_path, relative)
                tempname = relative[len(dataset_path) - 4:]
                aud_file = os.path.join(aud_dir, tempname)
                wav_fname = '_'.join(tempname.split('/'))
                ann_file = os.path.join(ann_dir, wav_fname.replace('.wav', '.beats'))

                # groundtruth
                beats_ann = np.loadtxt(ann_file)
                groundtruth_tempo_curve = tempoWin(beats_ann, win_len = 12, hop = 1)
                l = groundtruth_tempo_curve[6:]
                label.append(l)

                # melspectrogram
                audio, rate = librosa.load(aud_file)
                for i in range(len(l)):
                    # 12 seconds
                    audio_cut = audio[(i)*rate:(12+i)*rate]
                    S = librosa.feature.melspectrogram(y=audio_cut, sr=rate, n_mels=128,
                                                        fmax=8000, center = False)
                    S_dB = librosa.power_to_db(S, ref=np.max)
                    S_dB_norm = librosa.util.normalize(S_dB)
                    S_dB_transpose = S_dB_norm.transpose()
                    a.append(S_dB_transpose)
            
            features = np.array(a)
            flat_labels = [item for l in label for item in l]
            labels = np.array(flat_labels)
            speed = speed_detect(labels)
            np.save('/home/rich.tsai/NAS_189/home/BeatTracking/new/labels.npy', labels)
            np.save('/home/rich.tsai/NAS_189/home/BeatTracking/new/features.npy', features)
            np.save('/home/rich.tsai/NAS_189/home/BeatTracking/new/speed.npy', speed)

    X_train, X_valid, y_train, y_valid = \
    train_test_split(features, speed, test_size=0.2, random_state=42)
    
    date = '0808'
    exp_num = 1
    exp_name = 'RNNTempo_V'+str(exp_num) + '_' + date
    exp_dir = os.path.join('./experiments', exp_name)
    BATCH_SIZE = 8
    EPOCH = 2000
    LR = 0.001
    patience = 20

    train_loader = torch.utils.data.DataLoader(
        dataset = Beatdataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
        )
    
    valid_loader = torch.utils.data.DataLoader(
        dataset = Beatdataset(X_valid, y_valid),
        batch_size=BATCH_SIZE,
        shuffle=True
        )
    
    rnn = RNN()
    rnn.cuda(cuda_num)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_fun = nn.CrossEntropyLoss()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.3, 
            patience=80,
            cooldown=10
        )
    
    es = utils.EarlyStopping(patience = patience)
    
    valid_losses = []
    train_losses = []
    train_times = []
    lr_change_epoch = []
    best_epoch = 0
    stop_t = 0
    
    for epoch in range(1, EPOCH + 1):
    #    break
        end = time.time()
        train_loss = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x, b_y = b_x.to(device), b_y.to(device)
            r_out = rnn(b_x.float())
            loss = loss_fun(r_out, b_y.to(dtype =torch.long))

            optimizer.zero_grad()
            loss.backward()
            train_loss += loss
            optimizer.step()           
        train_loss = train_loss/len(train_loader.dataset)

        valid_loss = 0
        with torch.no_grad():
            for valid_x, valid_y in valid_loader:
                valid_x, valid_y = valid_x.to(device), valid_y.to(device)
                valid_out = rnn(valid_x.float())
                one_valid_loss = loss_fun(valid_out, valid_y.to(dtype=torch.long))
                valid_loss += one_valid_loss
            valid_loss = valid_loss/len(valid_loader.dataset)
            
        scheduler.step(valid_loss)
        train_losses.append(train_loss.item())
        valid_losses.append(valid_loss.item())
        
        stop = es.step(valid_loss.item())

        if valid_loss.item() == es.best:
            best_epoch = epoch
        
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': rnn.state_dict(),
            'best_loss': es.best,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
            },
            is_best=valid_loss == es.best,
            path=exp_dir,
            target='RNNTempoProc'
        )
        
        parameters = {
            'epochs_trained': epoch,
            'best_loss': es.best,
            'best_epoch': best_epoch,
            'train_loss_history': train_losses,
            'valid_loss_history': valid_losses,
            'train_time_history': train_times,
            'num_bad_epochs': es.num_bad_epochs,
            'lr_change_epoch': lr_change_epoch,
            'stop_t': stop_t,
            }
        
        with open(os.path.join(exp_dir, 'RNNTempo' + '.json'), 'w') as f:
            f.write(json.dumps(parameters, indent=4, sort_keys=True))
        
        train_times.append(time.time() - end)
        
        if stop:
            print("Early Stopping and Retrain")
            stop_t += 1
            if stop_t >= 5:
                break
            lr = LR*0.2
            lr_change_epoch.append(epoch)
            optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=0.3, 
                    patience=80,
                    cooldown=10
                )
            es = utils.EarlyStopping(patience= patience, best_loss = es.best)
#         e = [epoch, (valid_loss/len(valid_loader.dataset)).item()]
#         valid_losses.append(e)
                
#         one_loss = [epoch, (train_loss/len(train_loader.dataset)).item()]
#         train_losses.append(one_loss)

#                 acc = 0
#                 for valid_x, valid_y in valid_loader:
#                     valid_x = valid_x.to(device)
#                     valid_out = rnn(valid_x.float())
#                     pred_y = torch.max(valid_out, 1)[1].cpu().data.numpy()
#                     if pred_y == valid_y.numpy():
#                         acc += 1
                
#                 valid_acc = acc/len(valid_loader.dataset)
#                 e = [epoch, valid_acc]
#                 eval_valid.append(e)
                
#     np.save('/home/rich.tsai/NAS_189/home/BeatTracking/new/train_loss.npy', train_losses)          
#     np.save('/home/rich.tsai/NAS_189/home/BeatTracking/new/eval_list3.npy', eval_valid)


if __name__ == "__main__":
    main()