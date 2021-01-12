# import module
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib as jbl
import glob
import os

os.chdir("/data_st02/drug/inoue/CPI/classification/viz/viz_class5_cnn_lr0001_2/")

# 結果の統計表
file_list = sorted(glob.glob('*.jbl'))

file_num = []
target_label = []
true_label = []
prediction_score = []
sum_of_IG = []
seq_len = []
class_name = []
embedded_layer_IG = []
IGs_maxlen = []


l_class_name = ['classA', 'classB', 'classC', 'classD', 'classE'] 

for f in file_list:
    data = jbl.load(f)
    file_num.append(f.split('_')[1])
    target_label.append(data['target_label'])
    true_label.append(data['true_label'])
    prediction_score.append(data['prediction_score'])
    sum_of_IG.append(data['sum_of_IG'])
    seq_len.append(len(data['amino_acid_seq']))
    class_name.append(l_class_name[data['true_label']])
    IGs_maxlen.append(np.squeeze(data['embedded_layer_IG']).sum(axis=1))

col_list = [file_num, target_label, true_label,
            prediction_score, sum_of_IG, seq_len, class_name]
col_list_str = ['file_num', 'target_label', 'true_label',
                'prediction_score', 'sum_of_IG', 'seq_len', 'class_name']
data_mod = pd.DataFrame(data=col_list, index=col_list_str).T
data_mod.to_csv('./analize_class5.csv')


matrix_IGs_ = pd.DataFrame(IGs_maxlen, index=file_num).reset_index()
matrix_IGs = pd.concat([data_mod, matrix_IGs_],axis=1).sort_values('seq_len').iloc[:, 8:]
matrix_IGs_index = pd.concat([data_mod, matrix_IGs_],
                       axis=1).sort_values('seq_len').iloc[:, 5:7]
