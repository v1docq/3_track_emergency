import pandas as pd
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import LabelEncoder

seed = 42


def find_optimal_cutoff(target, pred_probas):
    fpr, tpr, threshold = roc_curve(target, pred_probas)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr)),
                        'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t['threshold'])


def analyze_threshold(model, X, y):
    th_probas_list = np.linspace(0, 1, num=100)
    precs = [];
    recalls = [];
    accs = [];
    f1s = []

    pred_probas = model.predict_proba(X)
    pred_probas_inv = np.array([[x[1], x[0]] for x in pred_probas])

    for th_proba in th_probas_list:
        preds = (pred_probas[:, 1] >= th_proba).astype('int')

        prec = precision_score(y, preds)
        recall = recall_score(y, preds)
        acc = accuracy_score(preds, y)
        f1 = f1_score(preds, y)

        precs.append(prec)
        recalls.append(recall)
        accs.append(acc)
        f1s.append(f1)

    # plot roc curve and get roc-auc
    pred_probas_moto = np.array([x[1] for x in pred_probas])
    pred_probas_drive = np.array([x[0] for x in pred_probas])
    fpr, tpr, thresholds = roc_curve(y, pred_probas_moto)
    plt.figure()
    plt.plot(fpr, tpr, marker='.')
    plt.title('ROC-curve')
    plt.xlabel('FRP')
    plt.ylabel('TPR')
    plt.show()

    # plot pr-curve
    pred_probas_moto = np.array([x[1] for x in pred_probas])
    prec, recall, _ = precision_recall_curve(y, pred_probas_moto)
    plt.figure()
    plt.plot(recall, prec, marker='.')
    plt.title('RP-curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

    # build curves as function of threshold
    plt.figure()
    plt.plot(th_probas_list, precs)
    plt.plot(th_probas_list, recalls)
    plt.plot(th_probas_list, accs)
    plt.plot(th_probas_list, f1s)
    plt.grid()
    plt.xlabel('moto proba threshold')
    plt.legend(['precision', 'recall', 'acc', 'f1s'])


def plot_feature_importance(importance, names, model_type, top_feat=20):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    fi_df = fi_df.iloc[:top_feat, :]
    # Define size of bar plot
    plt.figure(figsize=(10, 8))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


def load_train_test():
    df = pd.read_csv('./data_agg/new_train_enriched.csv', parse_dates=['datetime'])
    df = df.sort_values('datetime', ascending=True)
    df.reset_index(drop=True, inplace=True)
    df['bucket'] = round(df['road_km'] / 100., 0) * 100.
    # Create an object of the label encoder class
    labelencoder = LabelEncoder()

    # Apply labelencoder object on columns
    df['bucket'] = labelencoder.fit_transform(df['bucket'])
    df['hour'] = df['datetime'].apply(lambda x: x.hour)
    df['weekday'] = df['datetime'].apply(lambda x: x.weekday())
    df['year'] = df['datetime'].apply(lambda x: x.year)
    df['month'] = df['datetime'].apply(lambda x: x.month)

    test = pd.read_csv('./data_agg/new_test_enriched.csv', parse_dates=['datetime'])
    test['hour'] = test['datetime'].apply(lambda x: x.hour)
    test['weekday'] = test['datetime'].apply(lambda x: x.weekday())
    test['year'] = test['datetime'].apply(lambda x: x.year)
    test['month'] = test['datetime'].apply(lambda x: x.month)
    test['bucket'] = round(test['road_km'] / 100., 0) * 100.
    test['bucket'] = labelencoder.fit_transform(test['bucket'])

    road_9 = df.drop(df[df["road_id"].isin([5])].index)
    road_9.shape, Counter(road_9.road_id)

    condition1 = (road_9["target"] != 3)
    condition2 = (road_9['datetime'] > '2015-04-05')
    condition3 = (road_9['datetime'] < '2020-12-31')
    road_9 = road_9[(condition1 & condition2 & condition3)]

    return road_9, test


def load_agg_features(road_9, test, columns_filtred):
    # load features
    tele2_agg_join_by_roadid_weekday_month_hour_year_km = pd.read_csv(
        './data_agg/tele2_agg_join_by_roadid_weekday_month_hour_year_km.csv')
    speed_limits_join_roadid_km = pd.read_csv('./data_agg/speed_limits_join_roadid_km.csv')
    traffic_agg = pd.read_csv('./data_agg/traffic_agg_join_by_roadid_weekday_month_hour_year_km.csv')
    condition1 = (traffic_agg["road_id"].isin([9, 14]))
    traffic_agg = traffic_agg[condition1]

    # merged with train
    merged_1 = pd.merge(road_9, traffic_agg, on=['road_id', 'road_km', 'weekday', 'month', 'year', 'hour'], how='outer')
    merged_2 = pd.merge(merged_1, tele2_agg_join_by_roadid_weekday_month_hour_year_km,
                        on=['road_id', 'road_km', 'weekday', 'month', 'year', 'hour'], how='left')
    merged_3 = pd.merge(merged_2, speed_limits_join_roadid_km, on=['road_id', 'road_km'], how='left')
    merged_3['target'] = merged_3['target'].fillna(0)

    # merged with train
    tmp = merged_3
    tmp = tmp.fillna(0)
    tmp = tmp[columns_filtred]
    tmp.reset_index(drop=True, inplace=True)

    # split data to train and test
    tmp['datetime'] = tmp['datetime'].astype(str)
    condition1 = (tmp['year'] >= 2015)
    condition2 = (tmp['year'] < 2020)
    condition3 = (tmp['year'] >= 2020)
    tmp_train = tmp[(condition1 & condition2)]
    tmp_test = tmp[(condition3)]

    # get feature for test data
    test = pd.merge(test, traffic_agg, on=['road_id', 'road_km', 'weekday', 'month', 'year', 'hour'], how='left')
    test = test.drop_duplicates(['road_id', 'datetime', 'road_km'])
    test = test.fillna(0)

    return tmp, tmp_train, tmp_test, test


def get_features():
    cols = ['fixes_dummy_1',
            'fixes_dummy_2',
            'fixes_dummy_3',
            'fixes_dummy_4',
            'kmgr_0_st1',
            'kmgr_1_st1',
            'kmgr_2_st1',
            'kmgr_3_st1',
            'price_st1',
            'kmgr_0_st2',
            'kmgr_1_st2',
            'kmgr_2_st2',
            'kmgr_3_st2',
            'price_st2',
            'avg_volume',
            'min_volume',
            'avg_speed',
            'min_speed',
            'avg_occupancy',
            'min_occupancy',
            'max_occupancy',
            'datetime',
            'year',
            'month',
            'target']

    columns = [x for x in cols if x != 'target' and x != 'datetime' and x != 'year' and x != 'month']
    columns_filtred = cols

    return cols, columns, columns_filtred


if __name__ == '__main__':
    road_9, test = load_train_test()
    cols, columns, columns_filtred = get_features()
    tmp, tmp_train, tmp_test, test = load_agg_features(road_9, test, columns_filtred)

    tmp_train_x = tmp_train[columns]
    tmp_train_y = tmp_train['target']
    tmp_test_x = tmp_test[columns]
    tmp_test_y = tmp_test['target']

    clf = CatBoostClassifier(n_estimators=500,
                             learning_rate=0.05,
                             depth=10,
                             random_seed=seed,
                             objective='MultiClassOneVsAll',
                             bagging_temperature=0.2)

    clf.fit(tmp_train_x, tmp_train_y)
    predicted_target = clf.predict(tmp_test_x)
    print(f1_score(tmp_test_y, predicted_target, average='macro'))
    plot_feature_importance(clf.get_feature_importance(), tmp_train_x.columns, 'CATBOOST')

    test_x, test_y = test[columns], test['target']
    pred_stage1 = clf.predict_proba(test_x)

    # трешхолд для 1 класса отражает его частоту относительно нулевых эвентов
    # т.к. у нас смещенная мера вероятности из-за дисбаланса классов
    # то был выбран трешхолд который отражает "редкость" события в общем распределении
    class_1_threshold = 0.955
    class_2_threshold = 0.6

    preds_filtred = []
    for i in pred_stage1:
        if i[1] > class_1_threshold:
            preds_filtred.append(1)
        elif i[2] > class_2_threshold:
            preds_filtred.append(2)
        else:
            preds_filtred.append(0)

    pred_stage1 = pd.Series(preds_filtred)
    pred_stage1.index = test.index
    test['target'] = pred_stage1
    prediction = test[['datetime', 'road_id', 'road_km', 'target']]
    prediction.to_csv('./prediction_catboost.csv', index=False)
