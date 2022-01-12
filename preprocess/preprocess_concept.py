import numpy as np
import pandas as pd
import datatable as dt
from tqdm import tqdm

# Preprocess and make new train dataframe

NUM_PARTS = 7
NUM_TAGS_1 = 50
NUM_TAGS_2 = 100
NUM_TAGS_3 = 200


def importance_modifier(prev):
    return pow(prev, 5) * 20


train_data_types_dict = {
    'timestamp': 'int32',
    'user_id': 'int32',
    'content_id': 'int16',
    'content_type_id': 'bool',
    'task_container_id': 'int16',
    'answered_correctly': 'int8',
    'prior_question_elapsed_time': 'float32',
    'prior_question_had_explanation': 'bool'
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
                       nrows=30100000)
# train_df = train_df[train_df[target] != -1].reset_index(drop=True)
train_df['prior_question_had_explanation'].fillna(False, inplace=True)
train_df = train_df.astype(train_data_types_dict)

questions_df = pd.read_csv('../input/blonix-riiid-qtable-creation/q_table.csv')
questions_df = questions_df.astype(question_data_types_dict)
questions_df['importance_part'] = questions_df['importance_part'].apply(importance_modifier)
questions_df['importance_tags_1'] = questions_df['importance_tags_1'].apply(importance_modifier)
questions_df['importance_tags_2'] = questions_df['importance_tags_2'].apply(importance_modifier)
questions_df['importance_tags_3'] = questions_df['importance_tags_3'].apply(importance_modifier)
# questions_df['importance_tags_1'].plot.hist(bins=100, alpha=0.5)

train_grouped_user_df = train_df.groupby('user_id')
for name, group_df in tqdm(train_grouped_user_df, total=len(train_grouped_user_df)):  # iterate over each group
    user_score_correct = 0.5
    user_score_total = 1.0
    user_concept_parts_correct = [0.5] * NUM_PARTS
    user_concept_parts_total = [1.0] * NUM_PARTS
    user_concept_tags_1_correct = [0.5] * NUM_TAGS_1
    user_concept_tags_1_total = [1.0] * NUM_TAGS_1
    user_concept_tags_2_correct = [0.5] * NUM_TAGS_2
    user_concept_tags_2_total = [1.0] * NUM_TAGS_2
    user_concept_tags_3_correct = [0.5] * NUM_TAGS_3
    user_concept_tags_3_total = [1.0] * NUM_TAGS_3

    last_bundle_id = -1
    last_bundle_part = 0
    last_bundle_tags_1 = []
    last_bundle_tags_2 = []
    last_bundle_tags_3 = []
    user_score_bundle_correct = 0.0
    user_score_bundle_total = 0.0
    user_concept_parts_bundle_correct = 0.0
    user_concept_parts_bundle_total = 0.0
    user_concept_tags_1_bundle_correct = [0.0] * NUM_TAGS_1
    user_concept_tags_1_bundle_total = [0.0] * NUM_TAGS_1
    user_concept_tags_2_bundle_correct = [0.0] * NUM_TAGS_2
    user_concept_tags_2_bundle_total = [0.0] * NUM_TAGS_2
    user_concept_tags_3_bundle_correct = [0.0] * NUM_TAGS_3
    user_concept_tags_3_bundle_total = [0.0] * NUM_TAGS_3

    for index in group_df.index:  # for each data
        if group_df.at[index, target] != -1:
            id_index = group_df.at[index, 'content_id']

            question_bundle_id = questions_df.at[id_index, 'bundle_id']
            question_correctness = questions_df.at[id_index, 'correctness']
            question_part = questions_df.at[id_index, 'part']
            question_importance_part = questions_df.at[id_index, 'importance_part']
            question_tags_1 = questions_df.at[id_index, 'tags_1']
            question_tags_2 = questions_df.at[id_index, 'tags_2']
            question_tags_3 = questions_df.at[id_index, 'tags_3']
            question_importance_tags_1 = questions_df.at[id_index, 'importance_tags_1']
            question_importance_tags_2 = questions_df.at[id_index, 'importance_tags_2']
            question_importance_tags_3 = questions_df.at[id_index, 'importance_tags_3']

            if question_bundle_id != last_bundle_id:
                user_score_correct += user_score_bundle_correct
                user_score_total += user_score_bundle_total
                if last_bundle_part != 0:
                    user_concept_parts_correct[last_bundle_part - 1] += user_concept_parts_bundle_correct
                    user_concept_parts_total[last_bundle_part - 1] += user_concept_parts_bundle_total
                for last_tag in last_bundle_tags_1:
                    user_concept_tags_1_correct[last_tag] += user_concept_tags_1_bundle_correct[last_tag]
                    user_concept_tags_1_total[last_tag] += user_concept_tags_1_bundle_total[last_tag]
                    user_concept_tags_1_bundle_correct[last_tag] = 0.0
                    user_concept_tags_1_bundle_total[last_tag] = 0.0
                for last_tag in last_bundle_tags_2:
                    user_concept_tags_2_correct[last_tag] += user_concept_tags_2_bundle_correct[last_tag]
                    user_concept_tags_2_total[last_tag] += user_concept_tags_2_bundle_total[last_tag]
                    user_concept_tags_2_bundle_correct[last_tag] = 0.0
                    user_concept_tags_2_bundle_total[last_tag] = 0.0
                for last_tag in last_bundle_tags_3:
                    user_concept_tags_3_correct[last_tag] += user_concept_tags_3_bundle_correct[last_tag]
                    user_concept_tags_3_total[last_tag] += user_concept_tags_3_bundle_total[last_tag]
                    user_concept_tags_3_bundle_correct[last_tag] = 0.0
                    user_concept_tags_3_bundle_total[last_tag] = 0.0
                last_bundle_tags_1.clear()
                last_bundle_tags_2.clear()
                last_bundle_tags_3.clear()
                user_concept_parts_bundle_correct = 0.0
                user_concept_parts_bundle_total = 0.0

            last_bundle_id = question_bundle_id
            last_bundle_part = question_part
            last_bundle_tags_1.append(question_tags_1)
            last_bundle_tags_2.append(question_tags_2)
            last_bundle_tags_3.append(question_tags_3)

            train_df.at[index, 'user_score'] = user_score_correct / user_score_total  # cummulated user score
            train_df.at[index, 'concept_part'] = user_concept_parts_correct[question_part - 1] / \
                                                 user_concept_parts_total[
                                                     question_part - 1]  # concept understanding for part
            train_df.at[index, 'concept_tag_1'] = user_concept_tags_1_correct[question_tags_1] / \
                                                  user_concept_tags_1_total[
                                                      question_tags_1]  # concept understanding for tag
            train_df.at[index, 'concept_tag_2'] = user_concept_tags_2_correct[question_tags_2] / \
                                                  user_concept_tags_2_total[question_tags_2]
            train_df.at[index, 'concept_tag_3'] = user_concept_tags_3_correct[question_tags_3] / \
                                                  user_concept_tags_3_total[question_tags_3]
            # train_df.at[index, 'trivial_miss'] = user_score  # percentage of trivial miss
            # train_df.at[index, 'application'] = user_score  # application ability :
            # train_df.at[index, 'seriousness'] = user_score  # using lectures, check explanations ...

            user_score_bundle_total += (1.0 - question_correctness)
            user_concept_parts_bundle_total += question_importance_part
            user_concept_tags_1_bundle_total[question_tags_1] += question_importance_tags_1
            user_concept_tags_2_bundle_total[question_tags_2] += question_importance_tags_2
            user_concept_tags_3_bundle_total[question_tags_3] += question_importance_tags_3

            if group_df.at[index, target] == 1:  # if correct
                user_score_bundle_correct += (1.0 - question_correctness)
                user_concept_parts_bundle_correct += question_importance_part
                user_concept_tags_1_bundle_correct[question_tags_1] += question_importance_tags_1
                user_concept_tags_2_bundle_correct[question_tags_2] += question_importance_tags_2
                user_concept_tags_3_bundle_correct[question_tags_3] += question_importance_tags_3

train_df.to_csv('/kaggle/working/train_pre_concept.csv', sep=',', float_format='%.5f')