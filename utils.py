


__author__ = 'YaelSegal'
import soundfile
import librosa
import numpy as np
import torch
import torch.nn as nn
import os
from pathlib import Path
import time 
from datetime import timedelta
import subprocess
import numpy as np
import random
import math
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import textgrid
SIL=0
VOT=1
VOWEL=2
SR = 16000

def get_type(name):
    if "vowel" in name:
        return VOWEL
    if "vot" in name:
        return VOT
    return SIL

def get_name_by_type(ftype):
    if ftype == VOWEL:
        return "Vowel"
    if ftype == VOT:
        return "Vot"
    return ""
# Felix code!
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad
    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """

    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size).to(vec.device)], dim=dim)

def padd_list_tensors(targets, targets_lens, dim):

    target_max_len = max(targets_lens)
    padded_tensors_list = []
    for tensor, tensor_len in zip(targets, targets_lens):
        pad = pad_tensor(tensor, target_max_len, dim)
        padded_tensors_list.append(pad)
    padded_tensors = torch.stack(padded_tensors_list)
    return padded_tensors

def merge_close(sections_list):

    merge_section = [sections_list[0]]
    for index in range(1,len(sections_list)):
        prev_item = merge_section.pop()
        current_item = sections_list[index]
        if prev_item[2] == current_item[2]:
            merge_section.append([prev_item[0], current_item[1], current_item[2], current_item[1]-prev_item[0]])
        else:
            merge_section.append(prev_item)
            merge_section.append(current_item)

    return merge_section

def merge_type(sections_list, ftype):
    merge_section = [sections_list[0], sections_list[1]]
    for index in range(2,len(sections_list)):
        middle_item = merge_section.pop()
        first_item = merge_section.pop()
        last_item = sections_list[index]
        if middle_item[3]<15 and middle_item[2]==SIL and last_item[2]==ftype and first_item[2]==ftype:
            merge_section.append([first_item[0],last_item[1], ftype, first_item[3]+middle_item[3]+last_item[3]])
        else:
            merge_section.append(first_item)
            merge_section.append(middle_item)
            merge_section.append(last_item)

    return merge_section



def process_sections(preds_array, pre_process=False):
    change_value = np.diff(preds_array) 
    change_value_idx =  np.argwhere(change_value != 0)
    sections_list = []
    start_idx = 0
    for idx in change_value_idx:
        idx = idx[0]
        mark = preds_array[idx]
        item_len = idx - start_idx +1
        remove = False
        if get_name_by_type(mark) == get_name_by_type(VOT) and item_len<5 \
            or get_name_by_type(mark) == get_name_by_type(VOWEL) and item_len <20:
            # print("file:{},{} {}".format(new_filename, get_name_by_type(mark), start_idx / 1000))
            remove = True
 
        sections_list.append([start_idx, idx+1,mark, item_len, remove])
        start_idx = idx+1
    if start_idx != len(preds_array):
        sections_list.append([start_idx, len(preds_array)-1, 0, len(preds_array) - start_idx, False])
    if pre_process:
        new_sections_list = []
        for idx, (start_idx, end_idx, mark, item_len,remove) in enumerate(sections_list):
            if not remove:
                new_sections_list.append([start_idx, end_idx, mark, item_len])
                continue
            prev_item = sections_list[idx-1] if idx-1 >= 0 else None
            # next_item = sections_list[idx+1] if idx+1 <= len(sections_list) -1 else None
            new_sections_list.append([start_idx, end_idx, SIL, item_len])
            if prev_item and prev_item[2] == SIL:
                new_sections_list.pop()
                new_sections_list.append([prev_item[0], end_idx, SIL, end_idx- prev_item[0]])
        new_sections_list = merge_close(new_sections_list)
        new_sections_list = merge_type(new_sections_list, VOT)
        return new_sections_list
    else:
        return [x[:-1] for x in sections_list]

def create_textgrid(preds_array, new_filename, wav_len):

    sections_list = process_sections(preds_array, True)
    new_textgrid = textgrid.TextGrid()
    tier = textgrid.IntervalTier(name="preds", minTime=0)

    for item in sections_list:
        start_item, end_item, mark, item_len = item
        start_sec = start_item / 1000
        end_sec = end_item / 1000
        add = ""
        if get_name_by_type(mark) == get_name_by_type(VOT) and item_len <5:
            add = "short {}".format(item_len)
            print("file:{},vot {}:{}".format(new_filename, start_sec, end_sec))
        elif get_name_by_type(mark) == get_name_by_type(VOWEL) and item_len <30:
            add = "short {}".format(item_len)
            print("file:{},vowel {}:{}".format(new_filename, start_sec, end_sec))

        tier.add(start_sec, end_sec, get_name_by_type(mark) + add)


    new_textgrid.append(tier)
    new_textgrid.write(new_filename)

def sections2array(sections_list):
    array_len = sections_list[-1][1]
    new_array = np.zeros(array_len)
    for item in sections_list:
        new_array[item[0]: item[1]] = item[2]
    return new_array

def diff_accuracy(preds, targets, tolerance):
    eps = 1e-5
    preds_sections = process_sections(preds, True)
    new_pred_array = sections2array(preds_sections)
    # target_sections = process_sections(targets)
    change_value_pred =  np.where(np.diff(new_pred_array)!=0)
    # change_value_pred =  np.where(np.diff(preds)!=0)
    change_value_target = np.where(np.diff(targets)!=0)
    
    precision_counter = 0
    recall_counter = 0
    pred_counter = len(change_value_pred[0])
    gt_counter = len(change_value_target[0])
    if pred_counter !=0:
        for (y, yhat) in zip(change_value_target, change_value_pred):
            for yhat_i in yhat:
                diff = np.abs(y - yhat_i)
                min_dist = np.min(diff)
                min_dist_idx = np.argmin(diff)
                same_class = True
                if targets[change_value_target[0][min_dist_idx]] != new_pred_array[yhat_i]:
                # if targets[change_value_target[0][min_dist_idx]] != preds[yhat_i]:
                    same_class = False
                precision_counter += (min_dist <= tolerance) and same_class
            for y_i in y:
                diff = np.abs(yhat - y_i)
                min_dist = np.min(diff)
                min_dist_idx = np.argmin(diff)
                same_class = True
                if new_pred_array[change_value_pred[0][min_dist_idx]] != targets[y_i]:
                # if preds[change_value_pred[0][min_dist_idx]] != targets[y_i]:
                    same_class = False
                recall_counter += (min_dist <= tolerance) and same_class
    precision = precision_counter / (pred_counter + eps)
    recall = recall_counter / (gt_counter + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    print('tolerance: {}, precision:{}, recall:{}, f1:{}'.format(tolerance, precision, recall, f1))
    
    return precision, recall, f1


def diff_len_recall(preds, targets, tolerance_close, tolerance):
    eps = 1e-5
    preds_sections = process_sections(preds, True)
    new_pred_array = sections2array(preds_sections)
    # target_sections = process_sections(targets)
    change_value_pred =  np.where(np.diff(new_pred_array)!=0)
    # change_value_pred =  np.where(np.diff(preds)!=0)
    change_value_target = np.where(np.diff(targets)!=0)
    
    precision_counter = 0
    recall_counter = 0
    pred_counter = len(change_value_pred[0])
    gt_counter = len(change_value_target[0])
    vowel_counter_tol = 0
    vot_counter_tol = 0
    vowel_counter = 0
    vot_counter = 0
    gt_vowels = 0
    gt_vot = 0

    if pred_counter !=0:
        target_start = 0
        pred_start = 0
        for (y, yhat) in zip(change_value_target, change_value_pred):
            # for yhat_i in yhat:
            #     diff = np.abs(y - yhat_i)
            #     min_dist = np.min(diff)
            #     min_dist_idx = np.argmin(diff)
            #     same_class = True
            #     if targets[change_value_target[0][min_dist_idx]] != new_pred_array[yhat_i]:
            #     # if targets[change_value_target[0][min_dist_idx]] != preds[yhat_i]:
            #         same_class = False
            #     precision_counter += (min_dist <= tolerance) and same_class
            for y_i in y:
                diff = np.abs(yhat - y_i)
                min_dist = np.min(diff)
                min_dist_idx = np.argmin(diff)
                close_yhat = change_value_pred[0][min_dist_idx]
                same_class = True
                if min_dist <= tolerance_close:
                    if new_pred_array[close_yhat] == targets[y_i]:
                        len_target = y_i - target_start
                        len_pred = close_yhat - pred_start
                        diff_lens = abs(len_target - len_pred)
                        vowel_counter_tol += (diff_lens <= tolerance) and targets[y_i] == VOWEL
                        vot_counter_tol += (diff_lens <= tolerance) and targets[y_i] == VOT
                        vowel_counter += targets[y_i] == VOWEL
                        vot_counter += targets[y_i] == VOT
                pred_start = close_yhat +1
                target_start = y_i + 1

                gt_vowels +=  targets[y_i] == VOWEL
                gt_vot +=  targets[y_i] == VOT

    vowel_recall  = vowel_counter / (gt_vowels + eps)
    vot_recall = vot_counter / (gt_vot + eps)

    vowel_recall_tol  = vowel_counter_tol / (gt_vowels + eps)
    vot_recall_tol = vot_counter_tol / (gt_vot + eps)

    print('tolerance: {}, vowel recall:{} ,vowel tol:{} , vot recall:{}, vot tol:{}'.format(tolerance, vowel_recall, vowel_recall_tol, vot_recall, vot_recall_tol))
    
    return vowel_recall, vot_recall

def find_pairs(pred_sections, target_sections):
    pred_sections = np.array(pred_sections)
    target_sections = np.array(target_sections)
    pairs = []
    weird = []
    for idx, (pred_start, pred_end, ptype, pred_len) in enumerate(pred_sections):
        min_start_idx = np.abs(target_sections[:, 0]  - pred_start).argmin()
        min_end_idx = np.abs(target_sections[:, 1]  - pred_end).argmin()
        if min_start_idx == min_end_idx:
            pairs.append([idx, min_start_idx])
        else:
            weird.append(pred_sections[idx])
            one_target = target_sections[min_start_idx]
            two_target = target_sections[min_end_idx]
            un_one = min(one_target[1], pred_end) - max(one_target[0], pred_start)
            un_two = min(two_target[1], pred_end) - max(two_target[0], pred_start)
            if un_one> un_two and un_one > 0:
                pairs.append([idx, min_start_idx])
            elif un_two> un_one and un_two > 0:
                pairs.append([idx, min_end_idx])

    target_pairs = [x[1] for x in pairs]
    problem = []
    idx = 0
    while idx < len(pairs):
        pred_idx, target_idx = pairs[idx]
        current_problem = [idx]
        j = idx+1
        for next_idx in range(j, len(target_pairs)):
            if target_idx == target_pairs[next_idx]:
                current_problem.append(next_idx)
                idx+=1
        if len(current_problem) > 1:
            problem.append([target_idx,current_problem])
        idx+=1
    remove_list = []
    for target_idx, idx_list in problem:
        best_uni = - np.inf
        best_uni_idx = -1
        start_t = target_sections[target_idx][0]
        end_t = target_sections[target_idx][1]
        for idx in idx_list:
            pred_idx = pairs[idx][0]
            # pred_idx = idx
            start_p = pred_sections[pred_idx][0]
            end_p = pred_sections[pred_idx][1]
            uni = min(end_t, end_p) - max(start_t, start_p)
            if uni > best_uni:
                best_uni = uni
                best_uni_idx = idx
        idx_list.remove(best_uni_idx)
        remove_list.extend(idx_list)

    new_pairs = []
    for idx in range(len(pairs)):
        if idx not in remove_list:
            new_pairs.append(pairs[idx])

    return new_pairs

def actual_accuracy_tolerance(preds, targets, tolerance):
    eps = 1e-5
    preds_sections = process_sections(preds, True)
    target_sections = process_sections(targets)

    pred_vowels = [x for x in  preds_sections if x[2]==VOWEL]
    target_vowels = [x for x in  target_sections if x[2]==VOWEL]
    vowel_pairs = find_pairs(pred_vowels, target_vowels)

    vowel_tolerance = 0
    vowel_count = 0
    for pred_idx, target_idx in vowel_pairs:
        pred_item = pred_vowels[pred_idx]
        target_item = target_vowels[target_idx]
        if min(pred_item[1], target_item[1]) - max(pred_item[0], target_item[0]) < 0:
            continue
        if abs(pred_item[3] - target_item[3]) <= tolerance:
            vowel_tolerance += 1


    pred_vot = [x for x in  preds_sections if x[2]==VOT]
    target_vot = [x for x in  target_sections if x[2]==VOT]
    vot_pairs = find_pairs(pred_vot, target_vot)

    vot_tolerance = 0
    vot_count = 0
    for pred_idx, target_idx in vot_pairs:
        pred_item = pred_vot[pred_idx]
        target_item = target_vot[target_idx]
        if min(pred_item[1], target_item[1]) - max(pred_item[0], target_item[0]) < 0:
            continue
        if abs(pred_item[3] - target_item[3]) <= tolerance:
            vot_tolerance += 1
    
    # print("************* tolerance:{} ******************".format(tolerance))
    print('tolerance:{}, vowels recall:{}, vot recall:{}'.format(tolerance, vowel_tolerance/len(target_vowels), vot_tolerance/len(target_vot)))

def actual_accuracy_per(preds, targets, per_tolerance):
    eps = 1e-5
    preds_sections = process_sections(preds, True)
    target_sections = process_sections(targets)

    pred_vowels = [x for x in  preds_sections if x[2]==VOWEL]
    target_vowels = [x for x in  target_sections if x[2]==VOWEL]
    vowel_pairs = find_pairs(pred_vowels, target_vowels)

    vowel_tolerance = 0
    vowel_count = 0
    for pred_idx, target_idx in vowel_pairs:
        pred_item = pred_vowels[pred_idx]
        target_item = target_vowels[target_idx]
        current_per_tol = target_item[3] * 0.01 * per_tolerance
        if min(pred_item[1], target_item[1]) - max(pred_item[0], target_item[0]) < 0:
            continue
        if abs(pred_item[3] - target_item[3]) <= current_per_tol:
            vowel_tolerance += 1


    pred_vot = [x for x in  preds_sections if x[2]==VOT]
    target_vot = [x for x in  target_sections if x[2]==VOT]
    vot_pairs = find_pairs(pred_vot, target_vot)

    vot_tolerance = 0
    vot_count = 0
    for pred_idx, target_idx in vot_pairs:
        pred_item = pred_vot[pred_idx]
        target_item = target_vot[target_idx]
        current_per_tol = target_item[3] * 0.01 * per_tolerance
        if min(pred_item[1], target_item[1]) - max(pred_item[0], target_item[0]) < 0:
            continue
        if abs(pred_item[3] - target_item[3]) <= current_per_tol:
            vot_tolerance += 1
    
    # print("************* tolerance:{} ******************".format(tolerance))
    print('% tolerance: {}%, vowels recall:{}, vot recall:{}'.format(per_tolerance, vowel_tolerance/len(target_vowels), vot_tolerance/len(target_vot)))


def actual_accuracy(preds, targets):
    eps = 1e-5
    preds_sections = process_sections(preds, True)
    target_sections = process_sections(targets)

    pred_vowels = [x for x in  preds_sections if x[2]==VOWEL]
    target_vowels = [x for x in  target_sections if x[2]==VOWEL]
    vowel_pairs = find_pairs(pred_vowels, target_vowels)

    diffs_vowel = []
    for pred_idx, target_idx in vowel_pairs:
        pred_item = pred_vowels[pred_idx]
        target_item = target_vowels[target_idx]
        if min(pred_item[1], target_item[1]) - max(pred_item[0], target_item[0]) < 0:
            continue
        diffs_vowel.append(abs(target_item[-1] - pred_item[-1]))

    diffs_vowel = np.array(diffs_vowel)

    pred_vot = [x for x in  preds_sections if x[2]==VOT]
    target_vot = [x for x in  target_sections if x[2]==VOT]
    vot_pairs = find_pairs(pred_vot, target_vot)
    diffs_vot = []
    for pred_idx, target_idx in vot_pairs:
        pred_item = pred_vot[pred_idx]
        target_item = target_vot[target_idx]
        if min(pred_item[1], target_item[1]) - max(pred_item[0], target_item[0]) < 0:
            continue
        diffs_vot.append(abs(target_item[-1] - pred_item[-1]))

    diffs_vot = np.array(diffs_vot)

    vowel_precision = len(vowel_pairs)/(len(pred_vowels) + eps)
    vowel_recall = len(vowel_pairs)/(len(target_vowels) + eps)
    vot_precision = len(vot_pairs)/(len(pred_vot) + eps)
    vot_recall = len(vot_pairs)/(len(target_vot) + eps)
    print('vowels precision:{}, vowels recall:{}, vot precision:{}, vot recall:{}' \
    .format(vowel_precision, vowel_recall, \
    vot_precision, vot_recall))
    