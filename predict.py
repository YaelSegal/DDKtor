

__author__ = 'YaelSegal'

import torch.optim as optim
import torch
import torch.nn.functional as F
import argparse
import dataset 
import numpy as np
import os
import utils
from pathlib import Path
from model import CnnLstmRaw, load_model
import torch.nn as nn
import soundfile
import glob
import random
import math
from utils import SR, create_textgrid
import tqdm
# import librosa
import dataset
# import warnings
BUCH = '/data/segalya/ddk/UnsupSeg/buchwald_proccess_vot'


parser = argparse.ArgumentParser(description='test vowel/vot')
parser.add_argument('--data', type=str, default='./data/processed/' , help="BUCH ||",)
parser.add_argument('--out_dir', type=str, default='./data/out_tg/tmp_parts' , help="BUCH ||",)

# parser.add_argument('--data', type=str, default='/data/segalya/ddk/UnsupSeg/buchwald_proccess_vot/val' , help="BUCH ||",)
# parser.add_argument('--out_dir', type=str, default='/data/segalya/ddk/UnsupSeg/buchwald_proccess_vot/predict_textgrid' , help="BUCH ||",)


# parser.add_argument('--model', type=str, default='./model_cnn_lstm/data_BUCH_lr_0.0001_decay_1000_input_size_256_num_layers_2_hidden_size_256_channels_256_normalize_True_biLSTM_True_measure_loss_dropout_0.3_class_num_3_.pth', help='directory to save the model')

# new models
# parser.add_argument('--model', type=str, default='./model_cnn_lstm/data_BUCH_UPDATE_ntype_cnn_sim_lr_0.0001_input_size_256_num_layers_2_hidden_size_256_channels_256_normalize_True_norm_type_z_biLSTM_False_measure_rval_dropout_0.3_class_num_3_sigmoid_False_chain_bandreject,noise_lamda_1.0_59021734.pth', help='directory to save the model')
parser.add_argument('--model', type=str, default='./model_cnn_lstm/data_BUCH_UPDATE_ntype_lstm_sim_lr_0.0001_input_size_256_num_layers_2_hidden_size_256_channels_256_normalize_True_norm_type_z_biLSTM_True_measure_rval_dropout_0.3_class_num_3_sigmoid_False_chain_bandreject,noise_lamda_1.0_59021734.pth', help='directory to save the model')


parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--seed', type=int, default=1245,	help='random seed')


args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = 'cuda'
else:
    device = 'cpu'



########################################## testing ###########################################


path_model = args.model
test_model, normalize, sigmoid, norm_type  = load_model(path_model) 
if args.cuda:
    test_model = test_model.cuda()

with torch.no_grad():
    test_model.eval()
    files_list = glob.glob(args.data + "/*.wav")
    for wav_filename in tqdm.tqdm(files_list):
        wav_basename = wav_filename.split("/")[-1]

        # pred_dataset = dataset.PredictDataset(wav_filename,args.seed, slices_size=250, overlap=0, normalize=normalize, norm_type=norm_type)
        pred_dataset = dataset.PredictDataset(wav_filename,args.seed, slices_size=1000, overlap=0, normalize=normalize, norm_type=norm_type)
        pred_loader = torch.utils.data.DataLoader( pred_dataset, batch_size=100, shuffle=False,
        num_workers=0, pin_memory=args.cuda, collate_fn= dataset.PadCollatePred(dim=0))

        all_pred_class_idx = []
        conf = []
        for batch_idx, (raw, lens_list) in enumerate(pred_loader):
            raw = raw.to(device)

            hidden = test_model.init_hidden(raw.size(0), device)
            all_outputs = test_model(raw, hidden,lens_list)
            if len(all_outputs) == 2:
                output , hidden = all_outputs
            else:
                output, vector_out, hidden = all_outputs

            for idx in range(output.size(0)): 
                cur_len = lens_list[idx]
                # pred_class_idx = torch.argmax(output[idx,:cur_len], dim=1)
                pred_class_values, pred_class_idx = torch.max(output[idx,:cur_len], dim=1)
                conf.extend(F.softmax(output[idx,:cur_len], dim=1).cpu().numpy().tolist())
                pred_class_idx = pred_class_idx.cpu().numpy()
                pred_class_values = pred_class_values.cpu().numpy()
                all_pred_class_idx.extend(pred_class_idx)

        textgrid_basename = wav_basename.replace(".wav", ".TextGrid")
        textgrid_filename = os.path.join(args.out_dir, textgrid_basename)
        # create_textgrid(np.array(all_pred_class_idx), textgrid_filename, pred_dataset.wav_duration, conf)
        create_textgrid(np.array(all_pred_class_idx), textgrid_filename, pred_dataset.wav_duration)








