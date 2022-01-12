import pandas as pd
import warnings
from gensim import corpora,similarities,models
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')

TOPIC_NUM_1 = 50
TOPIC_NUM_2 = 100
TOPIC_NUM_3 = 200

question_data_types_dict = {
    'question_id': 'int32',
    'tags': 'object'
}
questions_df = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv', usecols=set(question_data_types_dict.keys()))
questions_df = questions_df.astype(question_data_types_dict)
questions_df['tags'].fillna("-1",inplace=True)

print('data loaded')

class tag_dic_c(object):
    def __iter__(self):
        for index,doc in enumerate(questions_df['tags']):
            yield doc.split(' ')
tag_dic = tag_dic_c()
dictionary=corpora.Dictionary(tag_dic)

print('dictionary done')

class vectorizer(object):
    def __iter__(self):
        for index,doc in enumerate(questions_df['tags']):
            yield dictionary.doc2bow(doc.split(' '))
corpus = vectorizer()
# corpus = [dictionary.doc2bow(text) for text in self.text]
tfidf_model = models.TfidfModel(corpus, id2word=dictionary)
corpus_tfidf = tfidf_model[corpus]

print('vectorize done')

lsi_model_1 = models.LsiModel(corpus_tfidf, id2word=dictionary,chunksize=2500000,num_topics=TOPIC_NUM_1)
lsi_model_2 = models.LsiModel(corpus_tfidf, id2word=dictionary,chunksize=2500000,num_topics=TOPIC_NUM_2)
lsi_model_3 = models.LsiModel(corpus_tfidf, id2word=dictionary,chunksize=2500000,num_topics=TOPIC_NUM_3)
#lsi_model.save("tag_lsi_model.lsi")
#lsi_model = models.LsiModel.load('tag_lsi_model.lsi')

print('Topic modeling done')


def get_arg_max(single_list):
    max_index=0
    max_num=single_list[0][1]
    for index in range(len(single_list)-1):
        if max_num<single_list[index+1][1]:
            max_num=single_list[index+1][1]
            max_index=index+1
    return max_index


all_data_lsi_1=[]
all_data_lsi_2=[]
all_data_lsi_3=[]

for tags in questions_df['tags']:
    vec_bow=dictionary.doc2bow(tags.split(' '))
    vec_lsi_1=list(lsi_model_1[tfidf_model[vec_bow]])
    vec_lsi_2=list(lsi_model_2[tfidf_model[vec_bow]])
    vec_lsi_3=list(lsi_model_3[tfidf_model[vec_bow]])
    if len(vec_lsi_1)==0:
        all_data_lsi_1.append(0)
    else:
        all_data_lsi_1.append(get_arg_max(vec_lsi_1))
    if len(vec_lsi_2)==0:
        all_data_lsi_2.append(0)
    else:
        all_data_lsi_2.append(get_arg_max(vec_lsi_2))
    if len(vec_lsi_3)==0:
        all_data_lsi_3.append(0)
    else:
        all_data_lsi_3.append(get_arg_max(vec_lsi_3))

questions_df['tags_lsi_1']=all_data_lsi_1
questions_df['tags_lsi_2']=all_data_lsi_2
questions_df['tags_lsi_3']=all_data_lsi_3

print(questions_df.head(50))
questions_df.drop(columns=['tags'], inplace=True)

print('Saving...')

questions_df.to_csv("tag_lsi.csv", sep=',')