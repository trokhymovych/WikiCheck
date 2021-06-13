import pickle5 as pickle
from copy import deepcopy
import json
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier, CatBoost, Pool, MetricVisualizer
from sklearn.model_selection import train_test_split

## loading model for building output of Model level two
from fairapi.modules.complex_model import WikiFactChecker
from fairapi.modules.logging_utils import *

with open('fever_end_test.pickle', 'rb') as handle:
    fever_test_set = pickle.load(handle)
with open('shared_task_dev.jsonl') as f:
    fever_actual = [json.loads(line) for line in f]
with open('train_set_labling.pickle', 'rb') as handle:
    train_agg = pickle.load(handle)

logger = get_logger(name=ROOT_LOGGER_NAME,
                    console=True,
                    log_level="INFO",
                    propagate=False)

config = {'bert_model_path': '/opt/math/jupyter/ntr/Experiments_with_text/models_filtered/bart_fine/bert_model_trained',
          'classification_model_path': '/opt/math/jupyter/ntr/Experiments_with_text/models_filtered/bart_fine/classifier_model',
          'cache_path': 'train_set_labling.pickle'}

model = WikiFactChecker(logger, **config)
results = []
for claim, lable in tqdm(train_agg.keys()):
    results.append((lable, model.predict_all_with_cache_agg((claim, lable))))

## Building train_dataset for classifier:
k = 10
features = []
lables = []
for lable, res in tqdm(results):
    try:
        a = pd.DataFrame(res)
        if lable == 'NOT ENOUGH INFO':
            a_n = np.random.choice(a.article.unique())
            a = a[a.article == a_n]
        a = a.sort_values('cos', ascending=False).head(k)

        if len(a) < k:
            a = pd.concat([a, pd.DataFrame(np.zeros((k - len(a), len(a.columns))), columns=a.columns)])
        features_local = list(a.cos) + list(a.contradiction_prob) + list(a.entailment_prob) + list(a.neutral_prob)
        features.append(features_local)
        lables.append(lable)
    except Exception as e:
        print(e)
        pass

results = []
for claim, lable in tqdm(train_agg.keys()):
    results.append((lable, model.predict_all_with_cache_agg((claim, lable))))

## Training a classifier model:
X_train, X_test, y_train, y_test = train_test_split(features, lables, test_size=0.1, random_state=1, shuffle=True,
                                                    stratify=lables)
model_clf = CatBoostClassifier(iterations=1000,
                               task_type="CPU")
model_clf = model_clf.fit(X_train,
                          y_train,
                          eval_set=(X_test, y_test),
                          verbose=False,
                          plot=True, use_best_model=True)

# Building Ranking dataset:
led = {'SUPPORTS': 1, 'REFUTES': 0}
with open('train_set_labling_ranking.pickle', 'rb') as handle:
    train_agg_ranking = pickle.load(handle)

def get_elem_from_list(claim, listik):
    for el in listik:
        if el[0] == claim:
            return el

features = []
lables = []
group = []
idd = 0
a = 0
for lable, res in tqdm(results):
    if lable != 'NOT ENOUGH INFO':
        res_tmp = deepcopy(res)

        claim = res[0]['claim']
        _, true_ev, _ = get_elem_from_list(claim, train_agg_ranking)

        tmp_evidence = dict()
        for el in true_ev:
            if el[1] is not None:
                article_ev = tmp_evidence.get(el[0], list())
                article_ev.append(el[1])
                tmp_evidence[el[0]] = article_ev

        for r in res_tmp:
            if r['keys'] in tmp_evidence.get(r['article'], []):
                r['is_evidence'] = 1
            else:
                r['is_evidence'] = 0

        a = pd.DataFrame(res_tmp).sort_values('cos', ascending=False)
        cos_min = np.min(a[a.is_evidence == 1].cos)
        a = a[a.cos >= cos_min - 0.2]

        for i, row in a.iterrows():
            features.append([row.cos, row.contradiction_prob, row.entailment_prob, row.neutral_prob, led[lable]])
            lables.append(row.is_evidence)
            group.append(idd)

        idd += 1

# training ranking model
test_ids = int(len(features) * 0.1)

train = Pool(
    data=features[:-test_ids],
    label=lables[:-test_ids],
    group_id=group[:-test_ids]
)

test = Pool(
    data=features[-test_ids:],
    label=lables[-test_ids:],
    group_id=group[-test_ids:]
)

default_parameters = {
    'iterations': 1500,
    'custom_metric': ['NDCG', 'PFound', 'AverageGain:top=10'],
    'verbose': False,
    'random_seed': 0,
}

parameters = {}


def fit_model(loss_function, additional_params=None, train_pool=train, test_pool=test):
    parameters = deepcopy(default_parameters)
    parameters['loss_function'] = loss_function
    parameters['train_dir'] = loss_function

    if additional_params is not None:
        parameters.update(additional_params)

    model = CatBoost(parameters)
    model.fit(train_pool, eval_set=test_pool, plot=True)

    return model
model_ranking = fit_model('YetiRankPairwise', {'custom_metric': ['PrecisionAt:top=5', 'RecallAt:top=5', 'MAP:top=5']})


# saving aggregation models:
a = {'clf_model': model_clf, 'ranking_model': model_ranking}
with open('aggregation_models.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
