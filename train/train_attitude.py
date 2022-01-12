import psutil
import os

pid = os.getpid()
current_process = psutil.Process(pid)

import numpy as np
import pandas as pd
import datatable as dt
import lightgbm as lgb
from matplotlib import pyplot as plt
# import riiideducation
# from tqdm import tqdm
import gc

_ = np.seterr(divide='ignore', invalid='ignore')

# preprocess

train_data_types_dict = {
    'user_id': 'int32',
    'content_id': 'int16',
    'answered_correctly': 'int8',
    'lagtime_1': 'float32',
    'lagtime_2': 'float32',
    'lagtime_3': 'float32',
    'user_explanation_mean': 'float16',
    'user_question_elapsed_time': 'float16',
    'all_attempt_n': 'int16',
    'lecture_n': 'int16',
    # 'lecture_time_delta': 'float32',
    'time_between_cluster': 'float32',
    # 'time_until_cluster': 'float32'
}
target = 'answered_correctly'
question_data_types_dict = {
    'question_id': 'int16',
    'part': 'int8',
    'tags_1': 'int8',
    'correctness': 'float16',
    'content_explanation_mean': 'float16',
    'part_bundle_id': 'int32',
    'content_elapsed_time_mean': 'float16',
    'content_score_true_mean': 'float16'
}
################################# Train Data Load ###################################

# train_df = dt.fread('../input/blonix-riiid-ttable-preprocess/t_table.csv', columns=set(train_data_types_dict.keys())).to_pandas()
train_df = pd.read_csv('../input/blonix-riiid-ttable-preprocess/t_table.csv', usecols=set(train_data_types_dict.keys()),
                       dtype=train_data_types_dict, nrows=10000000, skiprows=list(range(1, 30000000)))
train_df = train_df[train_df[target] != -1].reset_index(drop=True)
train_df = train_df.astype(train_data_types_dict)

current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2. ** 20
print(f"Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")

################################# Q Table Load & Merge ###################################

questions_df = pd.read_csv('../input/blonix-riiid-qtable-preprocess/q_table.csv',
                           usecols=set(question_data_types_dict.keys()))
questions_df = questions_df.astype(question_data_types_dict)

train_df = pd.merge(train_df, questions_df, left_on='content_id', right_on='question_id', how='left')
train_df.drop(columns=['question_id'], inplace=True)
del (questions_df)
gc.collect()

current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2. ** 20
print(f"Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")

################################# Train LGMB ###################################

features = [
    # 'all_attempt_n',
    # 'user_question_elapsed_time',
    # 'user_explanation_mean',

    'lagtime_1',
    'lagtime_2',
    'lagtime_3',
    # 'lecture_n',
    'time_between_cluster',

    'content_elapsed_time_mean',
    'content_score_true_mean',
    'content_explanation_mean',
    'part',
    # 'tags_1',
    # 'tags_2',
    # 'tags_3',
    'correctness'
]
# valid_df = train_df.groupby('user_id').tail(200)
valid_df = train_df.groupby('user_id').sample(frac=0.1)
train_df.drop(valid_df.index, inplace=True)

va_data = lgb.Dataset(valid_df[features], label=valid_df[target])
del (valid_df)
tr_data = lgb.Dataset(train_df[features], label=train_df[target])
del (train_df)
gc.collect()

current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2. ** 20
print(f"Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")

params = {
    # "boosting": 'dart'
    'feature_fraction': 0.70,  #
    # 'bagging_fraction': 0.50, #
    # 'bagging_freq': 10,

    'objective': 'binary',
    # 'two_round': True,
    'seed': 7734,
    'metric': 'auc',  # binary classification
    'learning_rate': 0.05,  # 0.05~0.1
    'max_bin': 500,  # default 255
    # 'max_depth': 13,  # default -1(no limit)
    'num_leaves': 1000
}

model = lgb.train(
    params,
    tr_data,
    num_boost_round=1000,
    valid_sets=[tr_data, va_data],
    early_stopping_rounds=50,
    verbose_eval=20
)

model.save_model(f'model_lgbm_attitude.txt')
lgb.plot_importance(model, importance_type='gain')
plt.show()