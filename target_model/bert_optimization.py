import torch
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from tqdm import tqdm as tqdm
from itertools import product
from bertopic import BERTopic
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import nltk
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
import os
import matplotlib.pyplot as plt
from pathlib import Path
from utils import load_anno_data

nltk.download('punkt')
nltk.download('wordnet')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_topic_model(n_neighbors, min_topic_size,calculate_probabilities = False):
    umap_model = UMAP(n_neighbors=n_neighbors, n_components=10, min_dist=0.0, metric='cosine')
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english", max_df=0.75,min_df=0.25)
    topic_model = BERTopic(umap_model=umap_model,vectorizer_model=vectorizer_model, calculate_probabilities=calculate_probabilities, verbose=True,min_topic_size=min_topic_size, nr_topics="auto")
    #topic_model.fit(documents)
    return topic_model
    
def get_coherence(df, topic_model):
    documents_per_topic = df.groupby(['Topic'], as_index=False).agg({'text': ' '.join})
    cleaned_docs = topic_model._preprocess_text(documents_per_topic.text.values)

    # Extract vectorizer and analyzer from BERTopic
    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()

    # Extract features for Topic Coherence evaluation
    words = vectorizer.get_feature_names()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words=[]
    for t in range(len(set(topics))-2):
        t_w=[]
        topic=topic_model.get_topic(t)
        if not isinstance(topic,bool):
            for words in topic:
                if words[0] not in tokens[0]:continue
                t_w.append(words[0])
            topic_words.append(t_w)
    
    # Evaluate
    coherence_model = CoherenceModel(topics=list(topic_words), 
                                     texts=tokens, 
                                     corpus=corpus,
                                     dictionary=dictionary, 
                                     coherence='c_v')
    coherence = coherence_model.get_coherence()
    return coherence

def percent(x):
    return str(round((x/sum_)*100,2))+'%'



# Load cleaned data
df = load_anno_data()
df['text'] = df.tokens.apply(lambda x: ' '.join(x))

n_neighbors = [15,20,25, 30, 35,40]
min_topic_sizes = [50, 100 ,150, 200, 250,300]

res = []
d = {'Number Of Negihbor':[],'Min Topic Size':[],'Num of Topic':[],'coherence':[],'Quantity of topic -1':[],'topic -1 %':[],'Total Amount':[]}
res_tabel = pd.DataFrame(data=d,index=[])


# temp2=[50,100,150,200]
# temp1=[15,20]

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.cuda.empty_cache()
index_save=0
for n_neighbor, min_topic_size  in tqdm(product(n_neighbors, min_topic_sizes), total=36):
    # if n_neighbors in temp1 and min_topic_size in temp2:
    #   print("continue")
    #   continue
   
    model = get_topic_model(n_neighbor, min_topic_size)
    topics, probas = model.fit_transform(df['text'])
    df['Topic']=topics
 

    print("finish transform")

    #Saving a data for this divition of topics
#    path_to_save='/home/roikreme/BertTopic/{}/random optimization/all/df_80k_trainer/{}_{}.pickle'.format(subreddit,n_neighbor,min_topic_size)
#    df.to_pickle(path_to_save,protocol=4)

    #Calaulate the amount of each topic
    get_topic=model.get_topic_info()
    c = Counter(topics)
    soret_c=sorted(c.items(),key=lambda x:x[0])
    count=[i[1] for i in soret_c]

    #update the dataframe with the ammount of each topic after transform
    get_topic['Count']=count

    sum_ = sum(count)
    get_topic['percentage']=get_topic['Count'].apply(percent)

    print("start coh")
    coh = get_coherence(df, model)

    #Adding to the dataframe
    res_tabel.loc[len(res_tabel.index)]=[str(n_neighbor),str(min_topic_size),str(len(get_topic)),coh,str(count[0]),get_topic['percentage'][0],str(sum_)]

    print("coh is:{}, min_topic_size:{}, n_neighbor:{} ".format(coh,min_topic_size,n_neighbor))
    #get_topic.to_csv("/home/roikreme/BertTopic/{}/random optimization/new_data/{}_{}_{}.csv".format(subreddit,n_neighbor,min_topic_size,coh),index=False)
   # pd.DataFrame(res).to_csv("/home/roikreme/BertTopic/random/{}/new_data/final_coh.csv".format(subreddit))
    
print("saving...")
res_tabel.to_csv(f'bert_optimization_res.csv')




 # create a plot for optimization - coherence
x=res_tabel['Num of Topic'].to_list()
y=res_tabel['coherence'].to_list()
plt.plot(x,y)
plt.xlabel('Num of Topic')
plt.ylabel('coherence')
plt.title('coherence graph')
n=[15,20,25]
m_topic=[50,100,150,200,250,300]
neg_topic=[(neg,topic) for neg,topic in product(n, m_topic)]

neg_topic=[str(r) for r in neg_topic]
score=[round(r['coherenc'],3) for r in res]

fig, axs = plt.subplots(1, figsize=(25, 10), sharey=True)

axs.set_xlabel("(n_neighbors, min_topic_sizes)")
axs.set_ylabel("coherence")

ymax=max(score)
xpos=score.index(ymax)
xmax=neg_topic[xpos]
axs.annotate("Max = {}".format(ymax),xy=(xmax,ymax),xytext=(xmax,ymax),arrowprops=dict(facecolor='black'))
axs.bar(neg_topic, score,width=0.5,align='center')
plt.savefig(f"coherence_plot.png")
