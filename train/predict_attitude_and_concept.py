import psutil
import os
pid = os.getpid()
current_process = psutil.Process(pid)

import numpy as np
import pandas as pd
import datatable as dt
import lightgbm as lgb
from matplotlib import pyplot as plt
#import riiideducation
#from tqdm import tqdm
import gc

_ = np.seterr(divide='ignore', invalid='ignore')

#preprocess

train_data_types_dict = {
    'user_id': 'int32',
    'content_id': 'int16',
    'answered_correctly': 'int8',
    'user_score': 'float16',
    'concept_part': 'float16',
    'concept_tag_1': 'float16',
    'concept_tag_2': 'float16',
    'concept_tag_3': 'float16',
    'lagtime_1': 'float32',
    'lagtime_2': 'float32',
    'lagtime_3': 'float32',
    'user_correctness': 'float16',
    'user_explanation_mean': 'float16',
    'user_question_elapsed_time': 'float16',
    'all_attempt_n': 'int16',
    'same_attempt_n': 'int8',
    'lecture_n': 'int16',
    'lecture_time_delta': 'float32',
    'time_between_cluster': 'float32',
    'time_until_cluster': 'float32'
}
target = 'answered_correctly'
question_data_types_dict = {
    'question_id': 'int16',
    'importance_part': 'float16',
    'importance_tags_1': 'float16',
    'importance_tags_2': 'float16',
    'importance_tags_3': 'float16'
}

################################# Train Data Load ###################################

#train_df = dt.fread('../input/blonix-riiid-ttable-preprocess/t_table.csv', columns=set(train_data_types_dict.keys())).to_pandas()
train_df = pd.read_csv('../input/blonix-riiid-ttable-preprocess/t_table.csv', usecols=set(train_data_types_dict.keys()), dtype=train_data_types_dict, nrows=20000000)
train_df = train_df[train_df[target] != -1].reset_index(drop=True)
train_df = train_df.astype(train_data_types_dict)

current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2.**20
print(f"Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")

################################# Q Table Load & Merge ###################################

questions_df = pd.read_csv('../input/blonix-riiid-qtable-preprocess/q_table.csv', usecols=set(question_data_types_dict.keys()))
questions_df = questions_df.astype(question_data_types_dict)

train_df = pd.merge(train_df, questions_df, left_on='content_id', right_on='question_id', how='left')
train_df.drop(columns=['question_id'], inplace=True)
del(questions_df)
gc.collect()

current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2.**20
print(f"Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")

################################# Last Simple Preprocess ###################################

train_df['concept_part'] = train_df['concept_part'] * train_df['importance_part']
train_df['concept_tag_1'] = train_df['concept_tag_1'] * train_df['importance_tags_1']
train_df['concept_tag_2'] = train_df['concept_tag_2'] * train_df['importance_tags_2']
train_df['concept_tag_3'] = train_df['concept_tag_3'] * train_df['importance_tags_3']
train_df['concept_part'] = train_df['concept_part'].astype('float16')
train_df['concept_tag_1'] = train_df['concept_tag_1'].astype('float16')
train_df['concept_tag_2'] = train_df['concept_tag_2'].astype('float16')
train_df['concept_tag_3'] = train_df['concept_tag_3'].astype('float16')
train_df.drop(columns=['importance_part', 'importance_tags_1', 'importance_tags_2', 'importance_tags_3'], inplace=True)

current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2.**20
print(f"Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")

################################# Predict Features ###################################

features_attitude = [
    'all_attempt_n',
    'user_question_elapsed_time',
    'user_explanation_mean',
    'lagtime_1',
    'lagtime_2',
    'lagtime_3',
    'lecture_n',
    'time_between_cluster'
]
model_attitude = lgb.Booster(model_file='../input/blonix-riiid-train-attitude/model_lgbm_attitude.txt')
train_df['attitude'] = model_attitude.predict(train_df[features_attitude])

features_concept = [
    'user_score',
    'concept_part',
    'concept_tag_1',
    'concept_tag_2',
    'concept_tag_3',
    'user_correctness'
]
model_concept = lgb.Booster(model_file='../input/blonix-riiid-train-concept/model_lgbm_concept.txt')
train_df['concept'] = model_concept.predict(train_df[features_concept])

train_df[['attitude']].to_csv('/kaggle/working/pretrain_attitude.csv', sep=',', na_rep='NaN', float_format = '%.5f')
train_df[['concept']].to_csv('/kaggle/working/pretrain_concept.csv', sep=',', na_rep='NaN', float_format = '%.5f')