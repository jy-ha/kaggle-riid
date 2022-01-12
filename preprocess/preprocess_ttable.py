import numpy as np
import pandas as pd
import datatable as dt
from tqdm import tqdm
import gc

_ = np.seterr(divide='ignore', invalid='ignore')

# preprocess

train_data_types_dict = {
    'timestamp': 'float32',
    'user_id': 'int32',
    'content_id': 'int16',
    'task_container_id': 'int16',
    'answered_correctly': 'int8',
    'content_type_id': 'int8',
    'prior_question_elapsed_time': 'float32',
    'prior_question_had_explanation': 'bool',
    'user_score': 'float16',
    'concept_part': 'float16',
    'concept_tag_1': 'float16',
    'concept_tag_2': 'float16',
    'concept_tag_3': 'float16'
}
target = 'answered_correctly'

# train_df = dt.fread('../input/blonix-riiid-data3/train_pre_concept.csv', columns=set(train_data_types_dict.keys())).to_pandas()
train_df = pd.read_csv('../input/blonix-riiid-data4/train_pre_concept.csv', usecols=set(train_data_types_dict.keys()),
                       dtype=train_data_types_dict, nrows=31000000)
train_df['prior_question_elapsed_time'].fillna(20000.0, inplace=True)
train_df = train_df.astype(train_data_types_dict)

print('Lecture Preprocess')

train_df['lecture_n'] = train_df[['user_id', 'content_type_id']].groupby(['user_id']).cumsum()['content_type_id']
train_df.lecture_n = train_df.lecture_n.astype('int16')

print('iter for each user-containers')
# print(train_df['timestamp'])
train_grouped_user_df = train_df.groupby('user_id')
for name, group_df in tqdm(train_grouped_user_df, total=len(train_grouped_user_df)):  # iterate over each group
    last_container_id = -1
    task_cnt = 0
    container_cnt = 0
    current_container_cnt = 0

    cluster_time_last = 0.0
    cluster_time_last_2 = 0.0
    lecture_time_last = 0.0
    lagtime_last = 0.0
    lagtime_1 = 0.0
    lagtime_2 = 0.0
    lagtime_3 = 0.0

    last_correctness_tot = 0.67
    last_explanation_mean_tot = 0.5
    last_question_elapsed_time_tot = 20000
    bundle_correctness_tot = 0.0
    bundle_explanation_mean_tot = 0.0
    bundle_question_elapsed_time_tot = 0.0

    for index in group_df.index:  # for each data
        time_stamp = group_df.at[index, 'timestamp']
        if group_df.at[index, 'content_type_id'] == 1:  # is lecture
            lecture_time_last = time_stamp
        else:
            container_id = group_df.at[index, 'task_container_id']
            task_cnt += 1

            if last_container_id != container_id:  # new container
                container_cnt += 1
                current_container_cnt = 1

                if (time_stamp - lagtime_last) > 12 * (1000 * 3600):  # 12hours
                    cluster_time_last_2 = cluster_time_last
                    cluster_time_last = time_stamp

                lagtime_3 = lagtime_2
                lagtime_2 = lagtime_1
                lagtime_1 = lagtime_last
                lagtime_last = time_stamp

                last_correctness_tot += bundle_correctness_tot
                last_explanation_mean_tot += bundle_explanation_mean_tot
                last_question_elapsed_time_tot += bundle_question_elapsed_time_tot
                bundle_explanation_mean_tot = group_df.at[index, 'prior_question_had_explanation']
                bundle_question_elapsed_time_tot = group_df.at[index, 'prior_question_elapsed_time']
                # bundle_correctness_tot = 0.0
                # bundle_explanation_mean_tot = 0.0
                bundle_question_elapsed_time_tot = 0.0
            else:
                current_container_cnt += 1
            last_container_id = container_id

            train_df.at[index, 'lecture_time_delta'] = (time_stamp - lecture_time_last) / (1000 * 3600)
            train_df.at[index, 'lagtime_1'] = lagtime_1
            train_df.at[index, 'lagtime_2'] = lagtime_2
            train_df.at[index, 'lagtime_3'] = lagtime_3
            train_df.at[index, 'time_between_cluster'] = cluster_time_last - cluster_time_last_2
            train_df.at[index, 'time_until_cluster'] = time_stamp - cluster_time_last
            bundle_correctness_tot += group_df.at[index, target]
            # bundle_explanation_mean_tot += group_df.at[index, 'prior_question_had_explanation']
            # bundle_question_elapsed_time_tot += group_df.at[index, 'prior_question_elapsed_time']

            if (task_cnt - current_container_cnt + 1) > 0:
                train_df.at[index, 'user_correctness'] = last_correctness_tot / (task_cnt - current_container_cnt + 1)
                train_df.at[index, 'user_question_elapsed_time'] = last_question_elapsed_time_tot / (
                            task_cnt - current_container_cnt + 1)
            else:
                train_df.at[index, 'user_correctness'] = last_correctness_tot
                train_df.at[index, 'user_question_elapsed_time'] = last_question_elapsed_time_tot

            train_df.at[index, 'user_explanation_mean'] = last_explanation_mean_tot / container_cnt

print('Question Preprocess')

train_df = train_df[train_df[target] != -1].reset_index(drop=True)
train_df.drop(columns=['content_type_id'], inplace=True)

# groupby.count() 로 변경 가능
train_df['all_attempt_n'] = 1
train_df.all_attempt_n = train_df.all_attempt_n.astype('int16')
train_df['all_attempt_n'] = train_df[['user_id', 'all_attempt_n']].groupby(['user_id'])['all_attempt_n'].cumsum()

train_df['same_attempt_n'] = 1
train_df.same_attempt_n = train_df.same_attempt_n.astype('int8')
train_df['same_attempt_n'] = train_df[['user_id', 'content_id', 'same_attempt_n']].groupby(['user_id', 'content_id'])[
    'same_attempt_n'].cumsum()

train_df['lagtime_1'] = train_df['timestamp'] - train_df['lagtime_1']
train_df['lagtime_2'] = train_df['timestamp'] - train_df['lagtime_2']
train_df['lagtime_3'] = train_df['timestamp'] - train_df['lagtime_3']
# lagtime_mean=train_df['lagtime_1'].mean()
# lagtime_mean2=train_df['lagtime_2'].mean()
# lagtime_mean3=train_df['lagtime_3'].mean()
# train_df['lagtime_1'].fillna(lagtime_mean, inplace=True)
# train_df['lagtime_2'].fillna(lagtime_mean2, inplace=True)
# train_df['lagtime_3'].fillna(lagtime_mean3, inplace=True)
train_df['lagtime_1'] = train_df['lagtime_1'] / (1000 * 3600)
train_df['lagtime_2'] = train_df['lagtime_2'] / (1000 * 3600)
train_df['lagtime_3'] = train_df['lagtime_3'] / (1000 * 3600)
train_df.lagtime_1 = train_df.lagtime_1.astype('float32')
train_df.lagtime_2 = train_df.lagtime_2.astype('float32')
train_df.lagtime_3 = train_df.lagtime_3.astype('float32')

train_df['time_between_cluster'] = train_df['time_between_cluster'] / (1000 * 3600)
train_df['time_until_cluster'] = train_df['time_until_cluster'] / (1000 * 3600)
train_df['time_between_cluster'] = train_df['time_between_cluster'].astype('float32')
train_df['time_until_cluster'] = train_df['time_until_cluster'].astype('float32')

train_df['lecture_time_delta'] = train_df['lecture_time_delta'].astype('float32')
train_df['user_correctness'] = train_df['user_correctness'].astype('float16')
train_df['user_explanation_mean'] = train_df['user_explanation_mean'].astype('float16')
train_df['user_question_elapsed_time'] = train_df['user_question_elapsed_time'].astype('float16')
train_df.drop(
    columns=['task_container_id', 'prior_question_had_explanation', 'prior_question_elapsed_time', 'timestamp'],
    inplace=True)

print(train_df)
train_df.to_csv('/kaggle/working/t_table.csv', sep=',', na_rep='NaN', float_format='%.5f')