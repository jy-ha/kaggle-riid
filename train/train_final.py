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
    'part': 'int8',
    # 'tags_1': 'int8',
    # 'tags_2': 'int8',
    # 'tags_3': 'int16',
    'importance_part': 'float16',
    'importance_tags_1': 'float16',
    'importance_tags_2': 'float16',
    'importance_tags_3': 'float16',
    'correctness': 'float16',
    'content_explanation_mean': 'float16',
    'part_bundle_id': 'int32',
    'content_elapsed_time_mean': 'float16',
    'content_score_false_mean': 'float16',
    'content_score_true_mean': 'float16'
}

################################# Train Data Load ###################################

# train_df = dt.fread('../input/blonix-riiid-ttable-preprocess/t_table.csv', columns=set(train_data_types_dict.keys())).to_pandas()
train_df = pd.read_csv('../input/blonix-riiid-ttable-preprocess/t_table.csv', usecols=set(train_data_types_dict.keys()),
                       dtype=train_data_types_dict, nrows=20000000)  # , skiprows=list(range(1,20000000)))
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

################################# Pre-train Load & Merge ###################################

pretrain_attitude_df = pd.read_csv('../input/blonix-riiid-pretrain/pretrain_attitude.csv', usecols=['attitude'])
pretrain_concept_df = pd.read_csv('../input/blonix-riiid-pretrain/pretrain_concept.csv', usecols=['concept'])
pretrain_attitude_df = pretrain_attitude_df.astype('float16')
pretrain_concept_df = pretrain_concept_df.astype('float16')

train_df = train_df.join(pretrain_attitude_df).join(pretrain_concept_df)
# train_df = pd.merge(train_df, pretrain_df, left_index=True, right_index=True)
del (pretrain_attitude_df, pretrain_concept_df)
gc.collect()

current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2. ** 20
print(f"Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")

################################# Last Simple Preprocess ###################################

train_df['content_score_difference'] = train_df['user_score'] - train_df['content_score_true_mean']
train_df['content_score_difference'] = train_df['content_score_difference'].astype('float16')
train_df.drop(columns=['content_score_true_mean', 'content_score_false_mean'], inplace=True)

train_df['concept_part'] = train_df['concept_part'] * train_df['importance_part']
train_df['concept_tag_1'] = train_df['concept_tag_1'] * train_df['importance_tags_1']
train_df['concept_tag_2'] = train_df['concept_tag_2'] * train_df['importance_tags_2']
train_df['concept_tag_3'] = train_df['concept_tag_3'] * train_df['importance_tags_3']
train_df['concept_part'] = train_df['concept_part'].astype('float16')
train_df['concept_tag_1'] = train_df['concept_tag_1'].astype('float16')
train_df['concept_tag_2'] = train_df['concept_tag_2'].astype('float16')
train_df['concept_tag_3'] = train_df['concept_tag_3'].astype('float16')
train_df.drop(columns=['importance_part'], inplace=True)

train_df['explanation_diff'] = train_df['user_explanation_mean'] - train_df['content_explanation_mean']

current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2. ** 20
print(f"Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")

################################# Train LGMB ###################################

features = [
    # 'concept',
    'user_score',  # weighted score
    'user_correctness',
    'concept_part',
    'concept_tag_1',
    'concept_tag_2',
    'concept_tag_3',
    # 'attitude',
    'all_attempt_n',
    'same_attempt_n',
    'user_question_elapsed_time',
    # 'user_explanation_mean',

    'content_elapsed_time_mean',
    # 'content_score_false_mean',
    # 'content_score_true_mean',
    'content_score_difference',
    'explanation_diff',
    # 'content_explanation_mean',
    'part',
    # 'tags_1',
    # 'tags_2',
    # 'tags_3',
    # 'importance_part',
    # 'importance_tags_1',
    # 'importance_tags_2',
    # 'importance_tags_3',
    'correctness',
    'lagtime_1',
    'lagtime_2',
    'lagtime_3',
    # 'lecture_n',
    'lecture_time_delta',
    # 'time_between_cluster',
    # 'time_until_cluster'
]
# print(train_df.loc[95, features])
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
    'feature_fraction': 0.50,  # 0.52
    'bagging_fraction': 0.50,  # 0.90
    # 'bagging_freq': 10,
    "boosting_type": "gbdt",
    "bagging_seed": 11,
    'reg_alpha': 0.3899927210061127,
    'reg_lambda': 0.6485237330340494,

    'objective': 'binary',
    # 'two_round': True,
    # 'seed': 7734,
    'metric': 'auc',  # for binary classification
    'learning_rate': 0.05,  # 크면 다른 파라미터 튜닝 편함. 0.05~0.1
    # 'num_iterations': 1000,  # default 100, num_boost_round 에서 이미 쓰임
    'max_bin': 500,  # 800 default 255, bigger better slower
    # 'max_depth': 13,  # default -1(no limit), 크면 오버핏
    'num_leaves': 1600  # max_depth 로 조절 가능, 동시에 쓰면 규제로 작용 1600
}

model = lgb.train(
    params,
    tr_data,
    num_boost_round=10000,
    valid_sets=[tr_data, va_data],
    early_stopping_rounds=50,
    verbose_eval=20
)

model.save_model(f'model_lgbm_blonix.txt')
lgb.plot_importance(model, importance_type='gain')
plt.show()