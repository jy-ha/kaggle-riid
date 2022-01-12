import numpy as np
import pandas as pd
import datatable as dt
from tqdm import tqdm

# Preprocess and make new Question Table

NUM_PARTS = 7
NUM_TAGS = 188

train_data_types_dict = {
    'user_id': 'int32',
    'content_id': 'int16',
    'answered_correctly': 'int8'
}
question_data_types_dict = {
    'question_id': 'int32',
    'bundle_id': 'int32',
    'part': 'int8',
    'importance_part': 'float16',
    'tags_1': 'int8',
    'tags_2': 'int8',
    'tags_3': 'int16',
    'importance_tags_1': 'float16',
    'importance_tags_2': 'float16',
    'importance_tags_3': 'float16',
    'correctness': 'float16'
}
target = 'answered_correctly'

# train_df = dt.fread('../input/riiid-test-answer-prediction/train.csv', columns=set(train_data_types_dict.keys())).to_pandas()
train_df = pd.read_csv('../input/riiid-test-answer-prediction/train.csv', usecols=set(train_data_types_dict.keys()),
                       nrows=20000000)
train_df = train_df[train_df[target] != -1].reset_index(drop=True)
train_df = train_df.astype(train_data_types_dict)

tag_lsi_df = pd.read_csv('../input/blonix-riiid-qtag-lsi/tag_lsi.csv')

questions_df = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')
questions_df.drop(columns=['correct_answer', 'tags'], inplace=True)
questions_df['tags_1'] = tag_lsi_df['tags_lsi_1']
questions_df['tags_2'] = tag_lsi_df['tags_lsi_2']
questions_df['tags_3'] = tag_lsi_df['tags_lsi_3']
questions_df['importance_part'] = 0.0
questions_df['importance_tags_1'] = 0.0
questions_df['importance_tags_2'] = 0.0
questions_df['importance_tags_3'] = 0.0
questions_df['correctness'] = train_df[target].groupby(train_df['content_id']).mean()
questions_df['correctness'].fillna(0, inplace=True)
questions_df = questions_df.astype(question_data_types_dict)

all_attampt_part_dict = dict()
all_attampt_tag_1_dict = dict()
all_attampt_tag_2_dict = dict()
all_attampt_tag_3_dict = dict()

for index in tqdm(questions_df.index, total=questions_df.shape[0]):  # for each question
    row_part = questions_df.at[index, 'part']
    row_tags_1 = questions_df.at[index, 'tags_1']
    row_tags_2 = questions_df.at[index, 'tags_2']
    row_tags_3 = questions_df.at[index, 'tags_3']
    row_question_id = questions_df.at[index, 'question_id']
    row_correctness = questions_df.at[index, 'correctness']

    all_attempt_q = train_df.loc[train_df.content_id == row_question_id]  # attempts for same question
    all_correct_people = all_attempt_q.loc[all_attempt_q[target] == 1, 'user_id']  # people who correct
    all_wrong_people = all_attempt_q.loc[all_attempt_q[target] == 0, 'user_id']  # people who wrong

    # calculate importance_part
    if row_part in all_attampt_part_dict:
        all_attampt_part = all_attampt_part_dict[row_part]
    else:
        all_question_same_part = questions_df.loc[
            questions_df['part'] == row_part, 'question_id']  # all questions in same part
        all_attampt_part = train_df.loc[
            train_df['content_id'].isin(all_question_same_part.values)]  # attempts for same part
        all_attampt_part_dict[row_part] = all_attampt_part

    part_correctness = all_attampt_part.loc[all_attampt_part.user_id.isin(all_correct_people.values), target]
    part_wrongness = all_attampt_part.loc[all_attampt_part.user_id.isin(all_wrong_people.values), target]
    # part_correctness = all_attampt_part.loc[all_attampt_part.join(all_correct_people.values, how='inner'), target]
    # part_wrongness = all_attampt_part.loc[[x in all_wrong_people.values for x in all_attampt_part.user_id], target]
    if part_correctness.empty:
        part_correctness = 0.0
    else:
        part_correctness = part_correctness.mean()

    if part_wrongness.empty:
        part_wrongness = 0.0
    else:
        part_wrongness = 1.0 - part_wrongness.mean()

    importance_part = (part_correctness * row_correctness) + (part_wrongness * (1.0 - row_correctness))

    # calculate importance_tags_1
    if row_tags_1 in all_attampt_tag_1_dict:
        all_attampt_tag = all_attampt_tag_1_dict[row_tags_1]
    else:
        all_question_same_tag = questions_df.loc[
            questions_df['tags_1'] == row_tags_1, 'question_id']  # all questions in same part
        all_attampt_tag = train_df.loc[
            train_df['content_id'].isin(all_question_same_tag.values)]  # attempts for same part
        all_attampt_tag_1_dict[row_tags_1] = all_attampt_tag

    tag_correctness = all_attampt_tag.loc[all_attampt_tag.user_id.isin(all_correct_people.values), target]
    tag_wrongness = all_attampt_tag.loc[all_attampt_tag.user_id.isin(all_wrong_people.values), target]

    if tag_correctness.empty:
        tag_correctness = 0.0
    else:
        tag_correctness = tag_correctness.mean()

    if tag_wrongness.empty:
        tag_wrongness = 0.0
    else:
        tag_wrongness = 1.0 - tag_wrongness.mean()

    importance_tag_1 = (tag_correctness * row_correctness) + (tag_wrongness * (1.0 - row_correctness))

    # calculate importance_tags_2
    if row_tags_2 in all_attampt_tag_2_dict:
        all_attampt_tag = all_attampt_tag_2_dict[row_tags_2]
    else:
        all_question_same_tag = questions_df.loc[
            questions_df['tags_2'] == row_tags_2, 'question_id']  # all questions in same part
        all_attampt_tag = train_df.loc[
            train_df['content_id'].isin(all_question_same_tag.values)]  # attempts for same part
        all_attampt_tag_2_dict[row_tags_2] = all_attampt_tag

    tag_correctness = all_attampt_tag.loc[all_attampt_tag.user_id.isin(all_correct_people.values), target]
    tag_wrongness = all_attampt_tag.loc[all_attampt_tag.user_id.isin(all_wrong_people.values), target]

    if tag_correctness.empty:
        tag_correctness = 0.0
    else:
        tag_correctness = tag_correctness.mean()

    if tag_wrongness.empty:
        tag_wrongness = 0.0
    else:
        tag_wrongness = 1.0 - tag_wrongness.mean()

    importance_tag_2 = (tag_correctness * row_correctness) + (tag_wrongness * (1.0 - row_correctness))

    # calculate importance_tags_3
    if row_tags_3 in all_attampt_tag_3_dict:
        all_attampt_tag = all_attampt_tag_3_dict[row_tags_3]
    else:
        all_question_same_tag = questions_df.loc[
            questions_df['tags_3'] == row_tags_3, 'question_id']  # all questions in same part
        all_attampt_tag = train_df.loc[
            train_df['content_id'].isin(all_question_same_tag.values)]  # attempts for same part
        all_attampt_tag_3_dict[row_tags_3] = all_attampt_tag

    tag_correctness = all_attampt_tag.loc[all_attampt_tag.user_id.isin(all_correct_people.values), target]
    tag_wrongness = all_attampt_tag.loc[all_attampt_tag.user_id.isin(all_wrong_people.values), target]

    if tag_correctness.empty:
        tag_correctness = 0.0
    else:
        tag_correctness = tag_correctness.mean()

    if tag_wrongness.empty:
        tag_wrongness = 0.0
    else:
        tag_wrongness = 1.0 - tag_wrongness.mean()

    importance_tag_3 = (tag_correctness * row_correctness) + (tag_wrongness * (1.0 - row_correctness))

    questions_df.at[index, 'importance_part'] = importance_part
    questions_df.at[index, 'importance_tags_1'] = importance_tag_1
    questions_df.at[index, 'importance_tags_2'] = importance_tag_2
    questions_df.at[index, 'importance_tags_3'] = importance_tag_3

questions_df.to_csv('/kaggle/working/q_table.csv', sep=',', float_format='%.5f')