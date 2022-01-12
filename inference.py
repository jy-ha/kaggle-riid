import numpy as np
import pandas as pd
import datatable as dt
from collections import defaultdict
import lightgbm as lgb
from matplotlib import pyplot as plt
import riiideducation
from tqdm import tqdm
import gc
import time

_ = np.seterr(divide='ignore', invalid='ignore')

train_data_types_dict = {
    'timestamp': 'int32',
    'task_container_id': 'int16',
    'user_id': 'int32',
    'content_id': 'int16',
    'content_type_id': 'int8',
    'answered_correctly': 'int8',
    'prior_question_elapsed_time': 'float32',
    'prior_question_had_explanation': 'int8'
}
target = 'answered_correctly'
question_data_types_dict = {
    'question_id': 'int16',
    'part': 'int8',
    'tags_1': 'int8',
    'tags_2': 'int8',
    'tags_3': 'int16',
    'importance_part': 'float16',
    'importance_tags_1': 'float16',
    'importance_tags_2': 'float16',
    'importance_tags_3': 'float16',
    'correctness': 'float16',
    'content_explanation_mean': 'float16',
    #'part_bundle_id': 'int32',
    'content_elapsed_time_mean': 'float32',
    #'content_score_false_mean': 'float16',
    'content_score_true_mean': 'float16'
}

#def importance_modifier(prev):
#    return pow((prev + 0.85) / 3, 5) * 20

print('Load Data...')

train_df = dt.fread('../input/riiid-test-answer-prediction/train.csv', columns=set(train_data_types_dict.keys())).to_pandas()
#train_df = pd.read_csv('../input/riiid-test-answer-prediction/train.csv', usecols=set(train_data_types_dict.keys()), dtype=train_data_types_dict, nrows=98)
#train_df = pd.read_csv('../input/riiid-test-answer-prediction/train.csv', usecols=set(train_data_types_dict.keys()), nrows=267)
#train_df = pd.read_csv('../input/riiid-test-answer-prediction/train.csv', usecols=set(train_data_types_dict.keys()), nrows=100, skiprows=list(range(1,100)))
train_df = train_df.groupby('user_id').tail(400)
train_df['prior_question_elapsed_time'].fillna(20000.0, inplace=True)
train_df['prior_question_had_explanation'].fillna(0, inplace=True)
train_df = train_df.astype(train_data_types_dict)

questions_df = pd.read_csv('../input/blonix-riiid-qtable-preprocess/q_table.csv', usecols=set(question_data_types_dict.keys()))
questions_df = questions_df.astype(question_data_types_dict)

#lecture_df = pd.read_csv("../input/riiid-test-answer-prediction/lectures.csv")

train_df = pd.merge(train_df, questions_df, left_on='content_id', right_on='question_id', how='left')
train_df.drop(columns=['question_id'], inplace=True)

print('Preprocess Data... lectures')
user_agg = train_df[['user_id', 'content_type_id', 'timestamp']].groupby(['user_id', 'content_type_id'])['timestamp'].agg(['max', 'count'])
user_agg=user_agg.unstack(fill_value=0)
user_agg=user_agg.reset_index()
user_agg.columns = ['user_id', 'timestamp_q', 'timestamp_l', 'count_q', 'count_l']
user_agg=user_agg.set_index('user_id')
u_lecture_n_dict = user_agg['count_l'].astype('int32').to_dict(defaultdict(int))
u_lecture_time_last_dict = user_agg['timestamp_l'].astype('float32').to_dict(defaultdict(int))
del(user_agg)
gc.collect()

train_df.drop(columns=['content_type_id'], inplace=True)
train_df = train_df[train_df[target] != -1].reset_index(drop=True)

print('Preprocess Data...1')
#train_df['importance_part'] = train_df['importance_part'].apply(importance_modifier)
concept_part_agg = train_df[['user_id', 'part', target, 'importance_part']].groupby(['user_id', 'part', target]).agg(['sum']).astype('float32')
concept_part_agg=concept_part_agg.unstack(fill_value=0.0)
concept_part_agg=concept_part_agg.reset_index()
concept_part_agg.columns = ['user_id', 'part', 'incorrect', 'correct']
concept_part_agg['total'] = concept_part_agg['incorrect'] + concept_part_agg['correct']
u_concept_parts_correct_dict = concept_part_agg.groupby(['user_id'])[['part', 'correct']].apply(lambda x: x.set_index('part')['correct'].to_dict(defaultdict(float))).to_dict(defaultdict(lambda: defaultdict(float)))
u_concept_parts_total_dict = concept_part_agg.groupby(['user_id'])[['part', 'total']].apply(lambda x: x.set_index('part')['total'].to_dict(defaultdict(float))).to_dict(defaultdict(lambda: defaultdict(float)))
train_df.drop(columns=['part', 'importance_part'], inplace=True)
del(concept_part_agg)
gc.collect()

print('Preprocess Data...2')
#train_df['importance_tags_1'] = train_df['importance_tags_1'].apply(importance_modifier)
#train_df['importance_tags_1'] = train_df['importance_tags_2'].apply(importance_modifier)
#train_df['importance_tags_1'] = train_df['importance_tags_3'].apply(importance_modifier)
concept_tag_1_agg = train_df[['user_id', 'tags_1', target, 'importance_tags_1']].groupby(['user_id', 'tags_1', target]).agg(['sum']).astype('float32')
concept_tag_2_agg = train_df[['user_id', 'tags_2', target, 'importance_tags_2']].groupby(['user_id', 'tags_2', target]).agg(['sum']).astype('float32')
concept_tag_3_agg = train_df[['user_id', 'tags_3', target, 'importance_tags_3']].groupby(['user_id', 'tags_3', target]).agg(['sum']).astype('float32')
concept_tag_1_agg=concept_tag_1_agg.unstack(fill_value=0.0)
concept_tag_2_agg=concept_tag_2_agg.unstack(fill_value=0.0)
concept_tag_3_agg=concept_tag_3_agg.unstack(fill_value=0.0)
concept_tag_1_agg=concept_tag_1_agg.reset_index()
concept_tag_2_agg=concept_tag_2_agg.reset_index()
concept_tag_3_agg=concept_tag_3_agg.reset_index()
concept_tag_1_agg.columns = ['user_id', 'tags_1', 'incorrect', 'correct']
concept_tag_2_agg.columns = ['user_id', 'tags_2', 'incorrect', 'correct']
concept_tag_3_agg.columns = ['user_id', 'tags_3', 'incorrect', 'correct']
concept_tag_1_agg['total'] = concept_tag_1_agg['incorrect'] + concept_tag_1_agg['correct']
concept_tag_2_agg['total'] = concept_tag_2_agg['incorrect'] + concept_tag_2_agg['correct']
concept_tag_3_agg['total'] = concept_tag_3_agg['incorrect'] + concept_tag_3_agg['correct']
u_concept_tags_1_correct_dict = concept_tag_1_agg.groupby(['user_id'])[['tags_1', 'correct']].apply(lambda x: x.set_index('tags_1')['correct'].to_dict(defaultdict(float))).to_dict(defaultdict(lambda: defaultdict(float)))
u_concept_tags_2_correct_dict = concept_tag_2_agg.groupby(['user_id'])[['tags_2', 'correct']].apply(lambda x: x.set_index('tags_2')['correct'].to_dict(defaultdict(float))).to_dict(defaultdict(lambda: defaultdict(float)))
u_concept_tags_3_correct_dict = concept_tag_3_agg.groupby(['user_id'])[['tags_3', 'correct']].apply(lambda x: x.set_index('tags_3')['correct'].to_dict(defaultdict(float))).to_dict(defaultdict(lambda: defaultdict(float)))
u_concept_tags_1_total_dict = concept_tag_1_agg.groupby(['user_id'])[['tags_1', 'total']].apply(lambda x: x.set_index('tags_1')['total'].to_dict(defaultdict(float))).to_dict(defaultdict(lambda: defaultdict(float)))
u_concept_tags_2_total_dict = concept_tag_2_agg.groupby(['user_id'])[['tags_2', 'total']].apply(lambda x: x.set_index('tags_2')['total'].to_dict(defaultdict(float))).to_dict(defaultdict(lambda: defaultdict(float)))
u_concept_tags_3_total_dict = concept_tag_3_agg.groupby(['user_id'])[['tags_3', 'total']].apply(lambda x: x.set_index('tags_3')['total'].to_dict(defaultdict(float))).to_dict(defaultdict(lambda: defaultdict(float)))
del(concept_tag_1_agg, concept_tag_2_agg, concept_tag_3_agg)
train_df.drop(columns=['tags_1', 'tags_2', 'tags_3', 'importance_tags_1', 'importance_tags_2', 'importance_tags_3'], inplace=True)
#del(concept_tag_1_agg)
gc.collect()

print('Preprocess Data...3')
train_df['score'] = 1 - train_df['correctness']
score_agg = train_df[['user_id', target, 'score']].groupby(['user_id', target]).agg(['sum']).astype('float32')
score_agg=score_agg.unstack(fill_value=0.0)
score_agg=score_agg.reset_index()
score_agg.columns = ['user_id', 'incorrect', 'correct']
score_agg['total'] = score_agg['incorrect'] + score_agg['correct']
u_score_correct_dict = score_agg[['user_id', 'correct']].set_index('user_id')['correct'].to_dict(defaultdict(float))
u_score_total_dict = score_agg[['user_id', 'total']].set_index('user_id')['total'].to_dict(defaultdict(float))
train_df.drop(columns=['score'], inplace=True)
del(score_agg)
gc.collect()

print('Preprocess Data...4')
same_attempt_n_agg = train_df[['user_id','content_id','answered_correctly']].groupby(['user_id','content_id'])['answered_correctly']
same_attempt_n_agg = same_attempt_n_agg.count().reset_index()
same_attempt_n_agg = same_attempt_n_agg.groupby(['user_id'])[['content_id', 'answered_correctly']]
u_same_attempt_n_dict = same_attempt_n_agg.apply(lambda x: x.set_index('content_id')['answered_correctly'].to_dict(defaultdict(int))).to_dict(defaultdict(lambda: defaultdict(int)))
del(same_attempt_n_agg)

print('Preprocess Data...5')
user_agg = train_df[['user_id', target]].groupby('user_id')[target].agg(['sum', 'count'])
content_agg = train_df[['content_id', target]].groupby('content_id')[target].agg(['sum', 'count'])
u_target_sum_dict = user_agg['sum'].astype('int32').to_dict(defaultdict(int))
u_target_cnt_dict = user_agg['count'].astype('int32').to_dict(defaultdict(int))
#q_target_sum_dict = content_agg['sum'].astype('int32').to_dict(defaultdict(int))
#q_target_cnt_dict = content_agg['count'].astype('int32').to_dict(defaultdict(int))
del(user_agg)
gc.collect()

print('Preprocess Data...6')
user_agg = train_df[['user_id', 'task_container_id', 'prior_question_had_explanation']].groupby(['user_id', 'task_container_id'])['prior_question_had_explanation'].agg(['sum']).reset_index()
user_agg['sum'] = user_agg['sum'].apply(lambda x: 1 if x>1 else x)
user_agg = user_agg.groupby('user_id')['sum'].agg(['sum']).reset_index()
u_explanation_sum_dict = user_agg['sum'].astype('int32').to_dict(defaultdict(int))
#content_agg = train_df.groupby('content_id')['prior_question_had_explanation'].agg(['sum'])
#q_explanation_sum_dict = content_agg['sum'].astype('int32').to_dict(defaultdict(int))
del(user_agg)
gc.collect()

print('Preprocess Data...7')
user_agg = train_df[['user_id', 'task_container_id', 'prior_question_elapsed_time']].groupby(['user_id', 'task_container_id'])['prior_question_elapsed_time'].agg(['sum', 'count']).reset_index()
user_agg['sum'] = user_agg['sum'] / user_agg['count']
user_agg = user_agg.groupby('user_id')['sum'].agg(['sum', 'count'])
#content_agg = train_df.groupby('content_id')['prior_question_elapsed_time'].agg(['sum', 'count'])
u_elapsed_time_sum_dict = user_agg['sum'].astype('float32').to_dict(defaultdict(float))
u_container_cnt_dict = user_agg['count'].astype('float32').to_dict(defaultdict(float))
#q_elapsed_time_sum_dict = content_agg['sum'].astype('float32').to_dict(defaultdict(float))
#q_elapsed_time_cnt_dict = content_agg['count'].astype('float32').to_dict(defaultdict(float))
#del(user_agg, content_agg)
del(user_agg)
gc.collect()

print('Preprocess Data...8')
train_df['lagtime'] = train_df[['user_id', 'timestamp']].groupby('user_id')['timestamp'].shift().fillna(0.0)
train_df['lagtime2'] = train_df[['user_id', 'timestamp']].groupby('user_id')['timestamp'].shift(2).fillna(0.0)
train_df['cluster_time'] = train_df['timestamp'] - train_df['lagtime']
train_df['cluster_time'] = train_df['cluster_time'].apply(lambda x: 1 if x>(12*(1000*3600)) else 0)
train_df['cluster_time'] = train_df['cluster_time'] * train_df['timestamp']
max_timestamp_cluster = train_df[['user_id', 'cluster_time']].groupby('user_id')['cluster_time'].agg(['max'])
max_timestamp_cluster_dict = max_timestamp_cluster['max'].astype('float32').to_dict(defaultdict(float))
max_timestamp_cluster_last_dict = defaultdict(float)
train_df.drop(columns=['cluster_time'], inplace=True)
del(max_timestamp_cluster)

max_timestamp_u = train_df[['user_id','timestamp']].groupby('user_id')['timestamp'].agg(['max'])
max_timestamp_u2 = train_df[['user_id','lagtime']].groupby('user_id')['lagtime'].agg(['max'])
max_timestamp_u3 = train_df[['user_id','lagtime2']].groupby('user_id')['lagtime2'].agg(['max'])
max_timestamp_u_dict = max_timestamp_u['max'].astype('float32').to_dict(defaultdict(float))
max_timestamp_u_dict2 = max_timestamp_u2['max'].astype('float32').to_dict(defaultdict(float))
max_timestamp_u_dict3 = max_timestamp_u3['max'].astype('float32').to_dict(defaultdict(float))
max_timestamp_u_dict4 = defaultdict(float)
train_df.drop(columns=['lagtime', 'lagtime2'], inplace=True)
del(max_timestamp_u, max_timestamp_u2, max_timestamp_u3)
del(train_df)
gc.collect()

print('Make Environment...')

features = [
    #'concept',
    #'attitude',
    'user_score',
    'user_correctness',
    'concept_part',
    'concept_tag_1',
    'concept_tag_2',
    'all_attempt_n',
    'same_attempt_n',
    'user_question_elapsed_time',

    'content_elapsed_time_mean',
    'content_score_difference',
    'explanation_diff',
    'part',
    'correctness',
    'lagtime_1',
    'lagtime_2',
    'lagtime_3',
    'lecture_n',
    'lecture_time_delta'
]
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
features_concept = [
    'user_score',
    'concept_part',
    'concept_tag_1',
    'concept_tag_2',
    'concept_tag_3',
    'user_correctness'
]

#model_attitude = lgb.Booster(model_file='../input/blonix-riiid-train-attitude/model_lgbm_attitude.txt')
#model_concept = lgb.Booster(model_file='../input/blonix-riiid-train-concept/model_lgbm_concept.txt')
model = lgb.Booster(model_file='../input/blonix-riiid-train-last/model_lgbm_blonix.txt')
env = riiideducation.make_env()
iter_test = env.iter_test()
prior_test_df = None

#time_last = time.time()
for t, (test_df, sample_prediction_df) in enumerate(iter_test):
    #time_start = time.time()
    #test_df = pd.read_csv("../input/riiid-test-answer-prediction/train.csv", nrows=10, skiprows=list(range(1,268)))
    #test_df = pd.read_csv("../input/riiid-test-answer-prediction/train.csv", nrows=10, skiprows=list(range(1,99)))
    #test_df = pd.read_csv("../input/riiid-test-answer-prediction/train.csv", nrows=10)
    #print(test_df.head(2))
    if prior_test_df is not None:
        prior_test_df[target] = eval(test_df['prior_group_answers_correct'].iloc[0])
        prior_test_df = prior_test_df[prior_test_df[target] != -1].reset_index(drop=True)
        last_container_id_prior = -1

        for row in prior_test_df[['user_id', 'content_id', target, 'part', 'tags_1', 'importance_part', 'importance_tags_1', 'correctness', 'tags_2', 'tags_3', 'importance_tags_2', 'importance_tags_3', 'task_container_id', 'prior_question_had_explanation','prior_question_elapsed_time']].values:
            if last_container_id_prior != row[12]:
                u_container_cnt_dict[row[0]] += 1.0
                u_explanation_sum_dict[row[0]] += row[13]
                u_elapsed_time_sum_dict[row[0]] += row[14]
            last_container_id_prior = row[12]

            u_target_sum_dict[row[0]] += row[2]
            u_target_cnt_dict[row[0]] += 1.0
            #q_target_sum_dict[content_id] += answered_correctly
            #q_target_cnt_dict[content_id] += 1

            u_score_total_dict[row[0]] += 1.0 - row[7]
            u_concept_parts_total_dict[row[0]][row[3]] += (row[5])
            u_concept_tags_1_total_dict[row[0]][row[4]] += (row[6])
            u_concept_tags_2_total_dict[row[0]][row[8]] += (row[10])
            u_concept_tags_3_total_dict[row[0]][row[9]] += (row[11])

            if row[2] == 1:
                u_score_correct_dict[row[0]] += 1.0 - row[7]
                u_concept_parts_correct_dict[row[0]][row[3]] += (row[5])
                u_concept_tags_1_correct_dict[row[0]][row[4]] += (row[6])
                u_concept_tags_2_correct_dict[row[0]][row[8]] += (row[10])
                u_concept_tags_3_correct_dict[row[0]][row[9]] += (row[11])

    test_df = pd.merge(test_df, questions_df, left_on='content_id', right_on='question_id', how='left')
    #test_df.drop(columns=['question_id'], inplace=True)
    test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].astype('float16').fillna(0.5)
    test_df['prior_question_elapsed_time'] = test_df['prior_question_elapsed_time'].fillna(20000.0)
    prior_test_df = test_df[['user_id', 'content_id', 'prior_group_answers_correct', 'part', 'tags_1', 'importance_part', 'importance_tags_1', 'correctness', 'tags_2', 'tags_3', 'importance_tags_2', 'importance_tags_3', 'task_container_id', 'prior_question_had_explanation','prior_question_elapsed_time']].copy()
    #prior_test_df = test_df[['user_id', 'content_id', 'part', 'tags_1','tags_2','tags_3', 'importance_part', 'importance_tags_1','importance_tags_2','importance_tags_3', 'correctness']].copy()

    question_len = len( test_df[test_df['content_type_id'] == 0])
    #question_len = len(test_df)

    u_lecture_n = np.empty(question_len, dtype=np.int32)
    u_lecture_time_delta = np.empty(question_len, dtype=np.float32)
    u_sum = np.empty(question_len, dtype=np.float32)
    u_count = np.empty(question_len, dtype=np.float32)
    #q_sum = np.empty(question_len, dtype=np.int32)
    #q_count = np.empty(question_len, dtype=np.int32)

    u_explanation_sum = np.empty(question_len, dtype=np.float32)
    #q_explanation_sum = np.empty(question_len, dtype=np.float16)
    u_elapsed_time_sum = np.empty(question_len, dtype=np.float32)
    u_container_cnt = np.empty(question_len, dtype=np.float32)
    #q_elapsed_time_sum = np.empty(question_len, dtype=np.float32)
    #q_elapsed_time_cnt = np.empty(question_len, dtype=np.float32)

    u_same_attempt_n =  np.empty(question_len, dtype=np.int8)
    u_score_total = np.empty(question_len, dtype=np.float32)
    u_score_correct  = np.empty(question_len, dtype=np.float32)
    u_concept_parts_total = np.empty(question_len, dtype=np.float32)
    u_concept_tags_1_total = np.empty(question_len, dtype=np.float32)
    u_concept_tags_2_total = np.empty(question_len, dtype=np.float32)
    u_concept_tags_3_total = np.empty(question_len, dtype=np.float32)
    u_concept_parts_correct = np.empty(question_len, dtype=np.float32)
    u_concept_tags_1_correct = np.empty(question_len, dtype=np.float32)
    u_concept_tags_2_correct = np.empty(question_len, dtype=np.float32)
    u_concept_tags_3_correct = np.empty(question_len, dtype=np.float32)

    lagtime_1 = np.empty(question_len, dtype=np.float32)
    lagtime_2 = np.empty(question_len, dtype=np.float32)
    lagtime_3 = np.empty(question_len, dtype=np.float32)
    u_time_between_cluster = np.empty(question_len, dtype=np.float32)
    u_time_until_cluster = np.empty(question_len, dtype=np.float32)

    last_container_id = -1
    #test_df.fillna(0, inplace=True)
    #print(test_df[['user_id', 'content_id', 'prior_question_elapsed_time']].head().values)
    lecture_cnt = 0
    for i, row in enumerate(test_df[['user_id', 'content_id', 'content_type_id', 'task_container_id', 'part', 'tags_1', 'prior_question_had_explanation','prior_question_elapsed_time', 'timestamp', 'tags_2', 'tags_3']].values):
        if row[2]==1:  # lecture1
            lecture_cnt += 1
            u_lecture_n_dict[row[0]] += 1
            u_lecture_time_last_dict[row[0]] = row[8]
        else:  # question
            if last_container_id != row[3]: # is new container
                if row[0] in max_timestamp_u_dict:
                    lagtime_1[i - lecture_cnt]=row[8]-max_timestamp_u_dict[row[0]]
                    if(row[0] in max_timestamp_u_dict2):#
                        lagtime_2[i - lecture_cnt]=row[8]-max_timestamp_u_dict2[row[0]]
                        if(row[0] in max_timestamp_u_dict3):
                            lagtime_3[i - lecture_cnt]=row[8]-max_timestamp_u_dict3[row[0]]
                            max_timestamp_u_dict4[row[0]]=max_timestamp_u_dict3[row[0]]
                            max_timestamp_u_dict3[row[0]]=max_timestamp_u_dict2[row[0]]
                        else:
                            lagtime_3[i - lecture_cnt]=0.0
                            max_timestamp_u_dict3[row[0]]=0.0
                        max_timestamp_u_dict2[row[0]]=max_timestamp_u_dict[row[0]]
                    else:
                        lagtime_2[i - lecture_cnt]=0.0
                        lagtime_3[i - lecture_cnt]=0.0
                        max_timestamp_u_dict2[row[0]]=0.0
                        max_timestamp_u_dict3[row[0]]=0.0
                    max_timestamp_u_dict[row[0]]=row[8]
                else:
                    lagtime_1[i - lecture_cnt]=0.0
                    lagtime_2[i - lecture_cnt]=0.0
                    lagtime_3[i - lecture_cnt]=0.0
                    max_timestamp_u_dict[row[0]]=row[8]
                    max_timestamp_u_dict2[row[0]]=0.0
                    max_timestamp_u_dict3[row[0]]=0.0
                if lagtime_1[i - lecture_cnt] > 12*(1000*3600):
                    max_timestamp_cluster_last_dict[row[0]] = max_timestamp_cluster_dict[row[0]]
                    max_timestamp_cluster_dict[row[0]] = row[8]
            else:
                lagtime_1[i - lecture_cnt]=row[8]-max_timestamp_u_dict2[row[0]]
                lagtime_2[i - lecture_cnt]=row[8]-max_timestamp_u_dict3[row[0]]
                lagtime_3[i - lecture_cnt]=row[8]-max_timestamp_u_dict4[row[0]]

            u_time_between_cluster[i - lecture_cnt] = max_timestamp_cluster_dict[row[0]] - max_timestamp_cluster_last_dict[row[0]]
            u_time_until_cluster[i - lecture_cnt] = row[8] - max_timestamp_cluster_dict[row[0]]

            u_same_attempt_n_dict[row[0]][row[1]] += 1
            #q_explanation_sum_dict[content_id] += explanation
            #q_elapsed_time_sum_dict[content_id] += elapsed_time
            #q_elapsed_time_cnt_dict[content_id] += 1

            u_lecture_n[i - lecture_cnt] = u_lecture_n_dict[row[0]]
            u_lecture_time_delta[i - lecture_cnt] = row[8] - u_lecture_time_last_dict[row[0]]
            u_sum[i - lecture_cnt] = u_target_sum_dict[row[0]]
            u_count[i - lecture_cnt] = u_target_cnt_dict[row[0]]
            #q_sum[i - lecture_cnt] = q_target_sum_dict[content_id]
            #q_count[i - lecture_cnt] = q_target_cnt_dict[content_id]

            u_same_attempt_n[i - lecture_cnt] = u_same_attempt_n_dict[row[0]][row[1]]
            u_explanation_sum[i - lecture_cnt] = u_explanation_sum_dict[row[0]]
            #q_explanation_sum[i - lecture_cnt] = q_explanation_sum_dict[content_id]
            u_elapsed_time_sum[i - lecture_cnt] = u_elapsed_time_sum_dict[row[0]]
            u_container_cnt[i - lecture_cnt] = u_container_cnt_dict[row[0]]
            #q_elapsed_time_sum[i - lecture_cnt] = q_elapsed_time_sum_dict[content_id]
            #q_elapsed_time_cnt[i - lecture_cnt] = q_elapsed_time_cnt_dict[content_id]

            u_score_total[i - lecture_cnt] = u_score_total_dict[row[0]]
            u_score_correct[i - lecture_cnt] = u_score_correct_dict[row[0]]
            u_concept_parts_total[i - lecture_cnt] = u_concept_parts_total_dict[row[0]][row[4]]
            u_concept_tags_1_total[i - lecture_cnt] = u_concept_tags_1_total_dict[row[0]][row[5]]
            u_concept_tags_2_total[i - lecture_cnt] = u_concept_tags_2_total_dict[row[0]][row[9]]
            u_concept_tags_3_total[i - lecture_cnt] = u_concept_tags_3_total_dict[row[0]][row[10]]
            u_concept_parts_correct[i - lecture_cnt] = u_concept_parts_correct_dict[row[0]][row[4]]
            u_concept_tags_1_correct[i - lecture_cnt] = u_concept_tags_1_correct_dict[row[0]][row[5]]
            u_concept_tags_2_correct[i - lecture_cnt] = u_concept_tags_2_correct_dict[row[0]][row[9]]
            u_concept_tags_3_correct[i - lecture_cnt] = u_concept_tags_3_correct_dict[row[0]][row[10]]

            last_container_id = row[3]

    test_df = test_df[test_df.content_type_id == 0]
    test_df['user_score'] = (u_score_correct+0.5) / (u_score_total+1.0)
    test_df['user_correctness'] = (u_sum+0.67) / (u_count+1)
    test_df['concept_part'] = ((u_concept_parts_correct+0.5) / (u_concept_parts_total+1.0)) * test_df['importance_part']
    test_df['concept_tag_1'] = ((u_concept_tags_1_correct+0.5) / (u_concept_tags_1_total+1.0)) * test_df['importance_tags_1']
    test_df['concept_tag_2'] = ((u_concept_tags_2_correct+0.5) / (u_concept_tags_2_total+1.0)) * test_df['importance_tags_2']
    #test_df['concept_tag_3'] = ((u_concept_tags_3_correct+0.5) / (u_concept_tags_3_total+1.0)) * test_df['importance_tags_3']
    test_df['all_attempt_n'] = u_count
    test_df['same_attempt_n'] = u_same_attempt_n
    test_df['user_question_elapsed_time'] = (u_elapsed_time_sum+20000) / (u_count+1.0)  #u_container_cnt??
    #test_df['user_explanation_mean'] = (u_explanation_sum+0.5) / (u_container_cnt+1.0)  #u_count??

    #test_df['content_elapsed_time_mean'] = q_elapsed_time_sum / q_elapsed_time_cnt
    test_df['content_score_difference'] = test_df['user_score'] - test_df['content_score_true_mean']
    #test_df['content_explanation_mean'] = q_explanation_sum / q_count# 1넘는게 있음
    test_df['explanation_diff'] = ((u_explanation_sum+0.5) / (u_count+1.0)) - test_df['content_explanation_mean']
    #test_df['correctness'] = q_sum / q_count
    test_df["lagtime_1"] = lagtime_1/(1000*3600)
    test_df["lagtime_2"] = lagtime_2/(1000*3600)
    test_df["lagtime_3"] = lagtime_3/(1000*3600)

    test_df["lecture_n"] = u_lecture_n
    test_df["lecture_time_delta"] = u_lecture_time_delta/(1000*3600)

    #test_df['time_between_cluster'] = u_time_between_cluster/(1000*3600)
    #test_df['time_until_cluster'] = u_time_until_cluster/(1000*3600)

    #test_df['attitude'] = model_attitude.predict(test_df[features_attitude])
    #test_df['concept'] = model_concept.predict(test_df[features_concept])
    #test_df.fillna(0, inplace=True)

    #print(test_df[features])
    #print(test_df[['user_id','user_score','user_correctness','user_question_elapsed_time','user_explanation_mean', 'time_between_cluster', 'time_until_cluster']])
    test_df[target] = model.predict(test_df[features])
    env.predict(test_df[['row_id', target]])

    #print('loop at %d :'%(t), test_df.shape, sample_prediction_df.shape)
    #print(time_start - time_last)
    #time_last = time_start
    #print(test_df.shape)

#time_run = time.time() - time_start
#print("run time : ", time_run)