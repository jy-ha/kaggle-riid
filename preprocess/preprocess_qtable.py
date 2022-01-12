import numpy as np
import pandas as pd
import datatable as dt
from tqdm import tqdm

_ = np.seterr(divide='ignore', invalid='ignore')

# preprocess

train_data_types_dict = {
    # 'timestamp': 'float32',
    'user_id': 'int32',
    'content_id': 'int16',
    'task_container_id': 'int16',
    'answered_correctly': 'int8',
    'prior_question_elapsed_time': 'float32',
    'prior_question_had_explanation': 'bool',
    'user_score': 'float32',
    # 'concept_part': 'float16',
    # 'concept_tag': 'float16'
}
target = 'answered_correctly'
question_data_types_dict = {
    'question_id': 'int16',
    'part': 'int8',
    'tags_1': 'int8',
    'tags_2': 'int8',
    'tags_3': 'int16',
    'bundle_id': 'int16',
    'importance_part': 'float16',
    'importance_tags_1': 'float16',
    'importance_tags_2': 'float16',
    'importance_tags_3': 'float16',
    'correctness': 'float16'
}


def importance_modifier(x):
    x = (x * 3) - 0.85
    if x < 0: x = 0
    return x


train_df = dt.fread('../input/blonix-riiid-data3/train_pre_concept.csv',
                    columns=set(train_data_types_dict.keys())).to_pandas()
# train_df = pd.read_csv('../input/blonix-riiid-data3/train_pre_concept.csv', usecols=set(train_data_types_dict.keys()), nrows=10000000, dtype=train_data_types_dict)
train_df = train_df[train_df[target] != -1].reset_index(drop=True)  # use only questions
train_df = train_df.astype(train_data_types_dict)

questions_df = pd.read_csv('../input/blonix-riiid-data3/q_table.csv', usecols=set(question_data_types_dict.keys()),
                           dtype=question_data_types_dict)

print('Start Preprocess')

questions_df['importance_part'] = questions_df['importance_part'].apply(importance_modifier)
questions_df['importance_tags_1'] = questions_df['importance_tags_1'].apply(importance_modifier)
questions_df['importance_tags_2'] = questions_df['importance_tags_2'].apply(importance_modifier)
questions_df['importance_tags_3'] = questions_df['importance_tags_3'].apply(importance_modifier)
# questions_df['importance_part'].plot.hist(bins=100, alpha=0.5)

# content_explanation_agg=train_df[['content_id','prior_question_had_explanation',target]].groupby(['content_id','prior_question_had_explanation'])[target].agg(['mean'])
# content_explanation_agg=content_explanation_agg.unstack()
# content_explanation_agg=content_explanation_agg.reset_index()
# content_explanation_agg.columns = ['content_id', 'content_explanation_false_mean','content_explanation_true_mean']
# content_explanation_agg.content_id=content_explanation_agg.content_id.astype('int16')
# content_explanation_agg.content_explanation_false_mean=content_explanation_agg.content_explanation_false_mean.astype('float16')
# content_explanation_agg.content_explanation_true_mean=content_explanation_agg.content_explanation_true_mean.astype('float16')
# questions_df = pd.merge(questions_df, content_explanation_agg, left_on='question_id', right_on='content_id', how='left')
# questions_df.drop(columns=['content_id'], inplace=True)
# del(content_explanation_agg)

questions_df['part_bundle_id'] = questions_df['part'] * 100000 + questions_df['bundle_id']
questions_df.part_bundle_id = questions_df.part_bundle_id.astype('int32')

content_score_agg = train_df[['content_id', target, 'user_score']].groupby(['content_id', target])['user_score'].agg(
    ['mean'])
content_score_agg = content_score_agg.unstack()
content_score_agg = content_score_agg.reset_index()
content_score_agg.columns = ['content_id', 'content_score_false_mean', 'content_score_true_mean']
content_score_agg.content_id = content_score_agg.content_id.astype('int16')
content_score_agg.content_score_false_mean = content_score_agg.content_score_false_mean.astype('float16')
content_score_agg.content_score_true_mean = content_score_agg.content_score_true_mean.astype('float16')
questions_df = pd.merge(questions_df, content_score_agg, left_on='question_id', right_on='content_id', how='left')
questions_df.drop(columns=['content_id'], inplace=True)
del (content_score_agg)

print('iter for each user-containers')

train_grouped_user_df = train_df.groupby('user_id')
for name, group_df in tqdm(train_grouped_user_df, total=len(train_grouped_user_df)):  # iterate over each group
    last_container_id = -1
    last_var_question_explanation = False
    last_var_question_elapsed_time = 20000.0

    for index in reversed(group_df.index):  # for each data reversed
        container_id = group_df.at[index, 'task_container_id']

        if last_container_id != container_id:  # new container
            now_var_question_explanation = last_var_question_explanation
            now_var_question_elapsed_time = last_var_question_elapsed_time
            last_var_question_explanation = group_df.at[index, 'prior_question_had_explanation']
            last_var_question_elapsed_time = group_df.at[index, 'prior_question_elapsed_time']
        train_df.at[index, 'question_explanation'] = now_var_question_explanation
        train_df.at[index, 'question_elapsed_time'] = now_var_question_elapsed_time
        last_container_id = container_id

train_df['question_explanation'] = train_df['question_explanation'].astype('int8')
questions_df['content_explanation_mean'] = \
train_df[['content_id', 'question_explanation']].groupby(['content_id'])['question_explanation'].agg(['mean'])['mean']
train_df.drop(columns=['question_explanation'], inplace=True)

questions_df['content_elapsed_time_mean'] = \
train_df[['content_id', 'question_elapsed_time']].groupby(['content_id'])['question_elapsed_time'].agg(['mean'])['mean']
train_df.drop(columns=['question_elapsed_time'], inplace=True)

print(questions_df)
questions_df.to_csv('/kaggle/working/q_table.csv', sep=',', na_rep='NaN', float_format='%.5f')