import time
import numba
import numpy as np
import pandas as pd
from scipy.stats import gmean
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import lightgbm as lgb

import config
from utils import woe, entropy2
from TargetEncoder import TargetEncoderWithThresh

#@author = "Mohsin"
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
LOGGER_FILE = 'model_trainer.log'
handler = logging.FileHandler(LOGGER_FILE)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
np.random.seed(0)

#Faster functions than pandas groupby aggregations :)
@numba.jit
def get_splits(a):
    m = np.concatenate([[True], a[1:] != a[:-1], [True]])
    m = np.flatnonzero(m)
    return m


@numba.jit
def get_expanding_count(user, time, col=None):
    """
    Takes sorted user_id and datetime columns, to give expanding count
    Optionally, take third column to apply function by both user_id and that column
    user: user_id array
    time: DateTime col (converted to int)
    col: attribute col (product/webpage etc)
    """
    out = np.zeros((len(user), ))
    if col is not None:
        col_unq = np.unique(col)
        for col_val in col_unq:
            col_val_idx = np.where(col == col_val)[0]
            col_user = user[col_val_idx]
            col_time = time[col_val_idx]
            col_out = np.zeros((len(col_val_idx), ))
            m = get_splits(col_user)
            n = len(m) -1
            for i in range(n):
                j = m[i]
                k = m[i+1]
                sub_time = col_time[j:k]
                oo = get_splits(sub_time)  #Only change count when datetime changes
                pp = len(oo) - 1
                for ii in range(pp):
                    col_out[j+oo[ii]:j+oo[ii+1]] = oo[ii] + 1
            out[col_val_idx] = col_out[:]
            
    else:
        m = get_splits(user) #Different user_ids
        n = len(m) -1
        for i in range(n):
            j = m[i]
            k = m[i+1]
            sub_time = time[j:k]
            oo = get_splits(sub_time) #Only change count when datetime changes
            pp = len(oo) - 1
            for ii in range(pp):
                out[j+oo[ii]:j+oo[ii+1]] = oo[ii] + 1
    return np.log1p(out)


@numba.jit
def get_prev_view(user, time, col=None):
    """
    Takes sorted user_id and datetime columns, to give delta between current Ad time and previous Ad time
    Optionally, take third column to apply function by both user_id and that column
    user: user_id array
    time: DateTime col (converted to int)
    col: attribute col (product/webpage etc)
    """
    out = -1*np.ones((len(user), ))
    if col is not None:
        col_unq = np.unique(col)
        for col_val in col_unq:
            col_val_idx = np.where(col == col_val)[0]
            col_user = user[col_val_idx]
            col_time = time[col_val_idx]
            col_out = -1*np.ones((len(col_val_idx), ))
            m = get_splits(col_user)
            n = len(m) -1
            for i in range(n):
                j = m[i]
                k = m[i+1]
                sub_time = col_time[j:k]
                oo = get_splits(sub_time) #Only change count when datetime changes (this is redundant here!!)
                pp = len(oo) - 1
                if pp == 0:
                    col_out[j] = 0
                else:
                    for ii in range(1, pp):
                        col_out[j+oo[ii]:j+oo[ii+1]] = np.log1p(col_time[j+oo[ii]] - col_time[j+oo[ii-1]])
            out[col_val_idx] = col_out[:]
            
    else:
        m = get_splits(user)
        n = len(m) -1
        for i in range(n):
            j = m[i]
            k = m[i+1]
            sub_time = time[j:k]
            oo = get_splits(sub_time)
            pp = len(oo) - 1
            if pp == 0:
                out[j] = -1
            else:
                for ii in range(1, pp):
                    out[j+oo[ii]:j+oo[ii+1]] = np.log1p(time[j+oo[ii]] - time[j+oo[ii-1]])
    return out


@numba.jit
def get_next_view(user, time, col=None):
    """
    Takes sorted user_id and datetime columns, to give delta between current Ad time and next Ad time
    Optionally, take third column to do apply function by both user_id and that column
    user: user_id array
    time: DateTime col (converted to int)
    col: attribute col (product/webpage etc)
    """
    out = -1*np.ones((len(user), ))
    if col is not None:
        col_unq = np.unique(col)
        for col_val in col_unq:
            col_val_idx = np.where(col == col_val)[0]
            col_user = user[col_val_idx]
            col_time = time[col_val_idx]
            col_out = -1*np.ones((len(col_val_idx), ))
            m = get_splits(col_user)
            n = len(m) -1
            for i in range(n):
                j = m[i]
                k = m[i+1]
                sub_time = col_time[j:k]
                oo = get_splits(sub_time)
                pp = len(oo) - 1
                if pp == 0:
                    col_out[j] = 0
                else:
                    for ii in range(pp-1):
                        col_out[j+oo[ii]:j+oo[ii+1]] = np.log1p(col_time[j+oo[ii+1]] - col_time[j+oo[ii]])
            out[col_val_idx] = col_out[:]
            
    else:
        m = get_splits(user)
        n = len(m) -1
        for i in range(n):
            j = m[i]
            k = m[i+1]
            sub_time = time[j:k]
            oo = get_splits(sub_time)
            pp = len(oo) - 1
            if pp == 0:
                out[j] = -1
            else:
                for ii in range(pp-1):
                    out[j+oo[ii]:j+oo[ii+1]] = np.log1p(time[j +oo[ii+1]] - time[j+oo[ii]])
    return out


@numba.jit
def get_click_counts(user, time, click, col=None):
    """
    Takes sorted user_id and datetime columns, and returns expanding count of Ad clicks
    Optionally, take third column to do apply function by both user_id and that column
    user: user_id array
    time: DateTime col (converted to int)
    click: Array for is_click
    col: attribute col (product/webpage etc)
    """
    out = np.zeros((len(user), ))
    if col is not None:
        col_unq = np.unique(col)
        for col_val in col_unq:
            col_val_idx = np.where(col == col_val)[0]
            col_user = user[col_val_idx]
            col_time = time[col_val_idx]
            col_click = click[col_val_idx]
            col_out = np.zeros((len(col_val_idx), ))
            m = get_splits(col_user)
            n = len(m) -1
            for i in range(n):
                cnt = 0
                j = m[i]
                k = m[i+1]
                sub_time = col_time[j:k]
                oo = get_splits(sub_time)
                pp = len(oo) - 1
                for ii in range(pp):
                    col_out[j+oo[ii]:j+oo[ii+1]] = cnt
                    cnt += np.sum(col_click[j+oo[ii]:j+oo[ii+1]])
            out[col_val_idx] = col_out[:]
            
    else:
        m = get_splits(user)
        n = len(m) -1
        for i in range(n):
            cnt = 0
            j = m[i]
            k = m[i+1]
            sub_time = time[j:k]
            oo = get_splits(sub_time)
            pp = len(oo) - 1
            for ii in range(pp):
                out[j+oo[ii]:j+oo[ii+1]] = cnt
                cnt += np.sum(click[j+oo[ii]:j+oo[ii+1]])
    return np.log1p(out)


def get_prev_click(user, time, click, col=None):
    """
    Takes sorted user_id and datetime columns, and returns delta between current time and last time Ad was clicked.
    Optionally, take third column to do apply function by both user_id and that column
    user: user_id array
    time: DateTime col (converted to int)
    click: Array for is_click
    col: attribute col (product/webpage etc)
    """
    out = -1* np.ones((len(user), ))
    if col is not None:
        col_unq = np.unique(col)
        for col_val in col_unq:
            col_val_idx = np.where(col == col_val)[0]
            col_user = user[col_val_idx]
            col_time = time[col_val_idx]
            col_click = click[col_val_idx]
            col_out = -1 * np.ones((len(col_val_idx), ))
            m = get_splits(col_user)
            n = len(m) -1
            for i in range(n):
                prev_time =-1
                j = m[i]
                k = m[i+1]
                sub_time = col_time[j:k]
                oo = get_splits(sub_time)
                pp = len(oo) - 1
                for ii in range(pp):
                    if prev_time != -1:
                        col_out[j+oo[ii]:j+oo[ii+1]] = np.log1p(col_time[j+oo[ii]] - prev_time)
                    if np.sum(col_click[j+oo[ii]:j+oo[ii+1]]) >= 1:
                        prev_time = col_time[j+oo[ii]] 
            out[col_val_idx] = col_out[:]
            
    else:
        m = get_splits(user)
        n = len(m) -1
        for i in range(n):
            prev_time = -1
            j = m[i]
            k = m[i+1]
            sub_time = time[j:k]
            oo = get_splits(sub_time)
            pp = len(oo) - 1
            for ii in range(pp):
                if prev_time != -1:
                    out[j+oo[ii]:j+oo[ii+1]] = np.log1p(time[j+oo[ii]] - prev_time)
                if np.sum(click[j+oo[ii]:j+oo[ii+1]]) >= 1:
                    prev_time = time[j+oo[ii]]
    return out


def get_view_feats(df):
    """
    Get all features related to when was previously or next time an Ad was shown to user,
    also combining with product, webpage and campaign_id
    :param df: DataFrame
    :return: DataFrame
    """
    df = df.copy()
    users = df["user_id"].values
    times = df["DateTime"].astype(int).values/10**12
    products = df["product"].values
    webpages = df["webpage_id"].values
    campaigns = df["campaign_id"].values
    df["prev_view"] = get_prev_view(users, times)
    
    logger.info("Getting previous product view")
    df["prev_prod_view"] = get_prev_view(users, times, products)
    df["prev_wp_view"] = get_prev_view(users, times, webpages)
    
    logger.info("Getting previous campaign view")
    df["prev_camp_view"] = get_prev_view(users, times, campaigns)
    df["next_view"] = get_next_view(users, times)
    df["next_prod_view"] = get_next_view(users, times, products)
    df["next_wp_view"] = get_next_view(users, times, webpages)
    df["next_camp_view"] = get_next_view(users, times, campaigns)    
    return df


def get_count_feats(df):
    """
    Get all expanding count features for user (how many times ad shown to user so far),
    also combining with product, webpage and campaign_id
    :param df: DataFrame
    :return: DataFrame
    """
    df = df.copy()
    uids = df["user_id"].values
    times = df["DateTime"].values
    products = df["product"].values
    wpids = df["webpage_id"].values
    cat1 = df["product_category_1"].values
    cat2 = df["product_category_2"].values
    df["view_counts"] = get_expanding_count(uids, times)
    df["prod_view_counts"] = get_expanding_count(uids, times, products)
    df["wp_view_counts"] = get_expanding_count(uids, times, wpids)
    df["cat1_view_counts"] = get_expanding_count(uids, times, cat1)
    df["cat2_view_counts"] = get_expanding_count(uids, times, cat2)
    return df


def get_click_feats(tr, val):
    """
    Get all features related to when was previously Ad clicked by user.
    also combining with product, webpage and campaign_id
    It is crucial to do this separately for training and test. as for test duration we can only map last information
    available from train time period
    :param tr: DataFrame
    :param val: DataFrame
    :return: (DataFrame, DataFrame)
    """
    tr = tr.copy()
    uids = tr["user_id"].values
    products = tr["product"].values
    wpids = tr["webpage_id"].values
    datetime = tr["DateTime"].astype(int).values/10**12
    clicks = tr["is_click"].values
    
    tr["prev_click"] = get_prev_click(uids, datetime, clicks)
    tr["prev_prod_click"] = get_prev_click(uids, datetime, clicks, products)
    tr["prev_wp_click"] = get_prev_click(uids, datetime, clicks, wpids)
    
    tr["click_counts"] = get_click_counts(uids, datetime, clicks)
    tr["prod_click_counts"] = get_click_counts(uids, datetime, clicks, products)
    tr["wp_click_counts"] = get_click_counts(uids, datetime, clicks, wpids)
    
    tr_clicks = tr.loc[tr.is_click == 1].groupby("user_id")["DateTime"].max()
    val["prev_click_time"] = val.user_id.map(tr_clicks)
    val["prev_click"] = np.log1p((val["DateTime"] - val["prev_click_time"]).astype(int)/10**12)
    val["prev_click"] = val["prev_click"].fillna(-1)
    del val["prev_click_time"]
    
    tr_clicks = tr.loc[tr.is_click == 1].groupby(["user_id", "product"])["DateTime"].max()
    tr_clicks.name = "prev_prod_click_time"
    val = val.join(tr_clicks, on=["user_id", "product"], how="left")
    val["prev_prod_click"] = np.log1p((val["DateTime"] - val["prev_prod_click_time"]).astype(int)/10**12)
    val["prev_prod_click"] = val["prev_prod_click"].fillna(-1)
    del val["prev_prod_click_time"]
                   
    tr_clicks = tr.loc[tr.is_click == 1].groupby(["user_id", "product"])["DateTime"].max()
    tr_clicks.name = "prev_wp_click_time"
    val = val.join(tr_clicks, on=["user_id", "product"], how="left")
    val["prev_wp_click"] = np.log1p((val["DateTime"] - val["prev_wp_click_time"]).astype(int)/10**12)
    val["prev_wp_click"] = val["prev_wp_click"].fillna(-1)
    del val["prev_wp_click_time"]
    
    val["click_counts"] = val["user_id"].map(np.log1p(tr.groupby("user_id")["is_click"].sum())).fillna(0)
    
    tmp = np.log1p(tr.groupby(["user_id", "product"])["is_click"].sum())
    tmp.name = "prod_click_counts"
    val = val.join(tmp, on=["user_id", "product"], how="left").fillna(0)
    
    tmp = np.log1p(tr.groupby(["user_id", "webpage_id"])["is_click"].sum())
    tmp.name = "wp_click_counts"
    val = val.join(tmp, on=["user_id", "webpage_id"], how="left").fillna(0)

    return tr, val


def get_overall_count_feats(df):
    """
    Combine tran and test and get overall counts by different combinations
    :param df:
    :return:
    """
    df = df.copy()
    enc = TargetEncoderWithThresh(cols=["user_id"], targetcol="is_click", func='count')
    df["all_counts"] = np.log1p(enc.fit_transform(df))

    enc = TargetEncoderWithThresh(cols=["product"], targetcol="is_click", func='count')
    df["prd_counts"] = np.log1p(enc.fit_transform(df))

    enc = TargetEncoderWithThresh(cols=["webpage_id"], targetcol="is_click", func='count')
    df["wp_counts"] = np.log1p(enc.fit_transform(df))

    enc = TargetEncoderWithThresh(cols=["user_id", "product"], targetcol="is_click", func='count')
    df["all_usr_prd_counts"] = np.log1p(enc.fit_transform(df))

    enc = TargetEncoderWithThresh(cols=["user_id", "webpage_id"], targetcol="is_click", func='count')
    df["all_usr_wp_counts"] = np.log1p(enc.fit_transform(df))

    enc = TargetEncoderWithThresh(cols=["user_group_id", "campaign_id"], targetcol="is_click", func='count')
    df["all_grp_camp_counts"] = np.log1p(enc.fit_transform(df))

    enc = TargetEncoderWithThresh(cols=["user_id", "product", "webpage_id"], targetcol="is_click", func='count')
    df["all_usr_prd_wp_counts"] = np.log1p(enc.fit_transform(df))

    enc = TargetEncoderWithThresh(cols=["product", "webpage_id"], targetcol="is_click", func='count')
    df["all_prd_wp_counts"] = np.log1p(enc.fit_transform(df))

    enc = TargetEncoderWithThresh(cols=["user_id", "DateTime"], targetcol="is_click", func='count')
    df["usr_date_counts"] = np.log1p(enc.fit_transform(df))

    enc = TargetEncoderWithThresh(cols=["user_id", "webpage_id", "DateTime"], targetcol="is_click", func='count')
    df["usr_wp_date_counts"] = np.log1p(enc.fit_transform(df))

    enc = TargetEncoderWithThresh(cols=["user_id", "DateTime"], targetcol="product", func='nunique')
    df["usr_date_nunq_prods"] = enc.fit_transform(df)
    return df


def get_target_encoding(tr, val, y_tr):
    """
    Get Target Encoding features for some of categorical variables and their combinations,
    We generate these features on train by cross-validation to avoid information leakage
    """
    cvlist2 = list(StratifiedKFold(10, shuffle=True, random_state=12345786).split(tr, y_tr))
    logger.info("Likelihood encoding product webpage_id")
    enc = TargetEncoderWithThresh(cols=["product", "webpage_id"], targetcol="is_click", use_prior=True, func='mean')
    tr["prod_wp_tmean"] = cross_val_predict(enc, tr, y_tr, cv=cvlist2, method="transform", n_jobs=-1)
    val["prod_wp_tmean"] = enc.fit(tr).transform(val)

    logger.info("Likelihood encoding product webpage_id")
    enc = TargetEncoderWithThresh(cols=["product", "webpage_id"], targetcol="user_id", use_prior=True, func=entropy2)
    tr["prod_wp_entropy"] = cross_val_predict(enc, tr, y_tr, cv=cvlist2, method="transform", n_jobs=-1)
    val["prod_wp_entropy"] = enc.fit(tr).transform(val)

    logger.info("Likelihood encoding user id")
    enc = TargetEncoderWithThresh(cols=["user_id"], targetcol="is_click", use_prior=True, func='mean', alpha=0.5)
    tr["usr_tmean"] = cross_val_predict(enc, tr, y_tr, cv=cvlist2, method="transform", n_jobs=-1)
    val["usr_tmean"] = enc.fit(tr).transform(val)

    logger.info("Likelihood encoding product and user group id")
    enc = TargetEncoderWithThresh(cols=["product", "user_group_id"], targetcol="is_click", use_prior=True, func='mean')
    tr["prod_grp_tmean"] = cross_val_predict(enc, tr, y_tr, cv=cvlist2, method="transform", n_jobs=-1)
    val["prod_grp_tmean"] = enc.fit(tr).transform(val)

    logger.info("Likelihood encoding product and user group id")
    enc = TargetEncoderWithThresh(cols=["user_depth", "age_level"], targetcol="is_click", use_prior=True, func='mean')
    tr["usr_age_tmean"] = cross_val_predict(enc, tr, y_tr, cv=cvlist2, method="transform", n_jobs=-1)
    val["usr_age_tmean"] = enc.fit(tr).transform(val)

    logger.info("Likelihood encoding product and user group id")
    enc = TargetEncoderWithThresh(cols=["user_depth", "age_level", "user_group_id"], targetcol="is_click", use_prior=True, func='mean')
    tr["usr_age_grp_tmean"] = cross_val_predict(enc, tr, y_tr, cv=cvlist2, method="transform", n_jobs=-1)
    val["usr_age_grp_tmean"] = enc.fit(tr).transform(val)

    logger.info("Mean of view counts for  product and webpage_id combination")
    enc = TargetEncoderWithThresh(cols=["product", "webpage_id"], targetcol="user_id", func='nunique')
    tr["prod_wp_unq_usr_enc"] = cross_val_predict(enc, tr, y_tr, cv=cvlist2, method="transform", n_jobs=-1)
    val["prod_wp_unq_usr_enc"] = enc.fit(tr).transform(val)

    return tr, val


def get_ratio_feats(df):
    """
    Derive some ratio's based on understanding of data
    :param df:
    :return:
    """
    df = df.copy()
    df["click_ratio"] = np.log1p(df["click_counts"]/df["view_counts"])
    df["clicks_done_ratio"] = np.log1p(df["view_counts"]/df["all_counts"])
    df["prev_next_rat"] = df["prev_view"]/(2+tr["next_view"])
    df["wp_click_ratio"] = np.log1p(df["wp_click_counts"]/df["wp_view_counts"])
    df["prod_adclick_ratio"] = np.log1p(df["usr_prod_adclicks"]/(1+np.log1p(df['total_adclicks'])))
    df["curr_prod_count_rat"] = np.log1p(df["prod_view_counts"]/df["all_usr_prd_counts"])
    df["all_prd_cnts"] = df["prod_click_counts"] + df["usr_prod_adclicks"]
    return df


def get_extra_feats(df):
    """
    Use counts on different products and dayweek by user as a feature (This can help cluster similar users together)
    :param df:
    :return:
    """
    df = df.copy()
    tmp = df.groupby(["user_id", "dayofweek"]).size()
    tmp.name = "usr_day_curr_counts"
    df = df.join(tmp, on=["user_id", "dayofweek"], how="left")

    tmp = df.groupby(["user_id", "product", "dayofweek"]).size()
    tmp.name = "usr_prd_day_curr_counts"
    df = df.join(tmp, on=["user_id", "product", "dayofweek"], how="left")

    tmp = df.groupby(["product", "webpage_id", "dayofweek"]).size()
    tmp.name = "prd_wp_day_curr_counts"
    df = df.join(tmp, on=["product", "webpage_id", "dayofweek"], how="left")

    tmp = df.groupby(["user_id", "product", "webpage_id", "dayofweek"]).size()
    tmp.name = "usr_prd_wp_day_curr_counts"
    df = df.join(tmp, on=["user_id", "product", "webpage_id", "dayofweek"], how="left")
    logger.info("Done day of week aggregations")

    tmp = df.groupby(["user_id", "product"]).size().unstack()
    tmp.columns = [f"usr_prd_other_views_{i}" for i in range(10)]
    df = df.join(tmp, on=["user_id"], how="left")

    tmp = df.groupby(["user_id", "webpage_id"]).size().unstack()
    tmp.columns = [f"usr_wp_other_views_{i}" for i in range(9)]
    df = df.join(tmp, on=["user_id"], how="left")
    logger.info("Done other product and webpage features on current data")
    return df


def get_test_preds(df, feats, lgb_params, flag=1):
    if flag == 1:
        train = df.loc[(df["DateTime"] < pd.to_datetime("2017/07/08"))].reset_index()
        test = df.loc[df["DateTime"] >= pd.to_datetime("2017/07/08")].reset_index()
    else:
        train = df.loc[(df["DateTime"] < pd.to_datetime("2017/07/08")) & (df["DateTime"] >= pd.to_datetime("2017/07/03"))].reset_index()
        test = df.loc[df["DateTime"] >= pd.to_datetime("2017/07/08")].reset_index()

    logger.info(f"Train and Test shapes : {train.shape, test.shape}")
    y_tr = train["is_click"].values

    train, test = get_click_feats(train, test)
    train, test = get_target_encoding(train, test, y_tr)
    train = get_ratio_feats(train)
    test = get_ratio_feats(test)

    #for f in feats:
    #    sns.distplot(train[f].fillna(0).replace(np.inf, -1))
    #    sns.distplot(test[f].fillna(0).replace(np.inf, -1))
    #    plt.show()

    X_tr, X_test = train[feats], test[feats]
    X_tr = X_tr.fillna(0)
    X_test = X_test.fillna(0)

    test_preds = []
    for seed in [12345786, 45, 101]:
        #lgb_params["seed"] = seed   #Bag multiple runs to make it easier to reproduce stuff (forgot to update in LB solution)
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr)], eval_metric='auc', verbose=50)
        test_preds.append(model.predict_proba(X_test)[:, 1])
    test_preds = gmean(test_preds, axis=0)
    return model, test_preds


if __name__ == "__main__":
    start_time = time.time()
    t = time.time()
    #>>>>>>>>>>>>>>>>Read data
    train = pd.read_csv(str(config.DATA_PATH / "train.csv"), parse_dates=["DateTime"])
    test = pd.read_csv(str(config.DATA_PATH / "test.csv"), parse_dates=["DateTime"])

    train["flag1"] = "train"
    test["flag1"] = "test"
    test["is_click"] = 0

    data_all = pd.concat([train, test], axis=0).reset_index()
    del test
    logger.info(f"Data shapes for train and combined train+test: {train.shape, data_all.shape}")


    #>>>>>>>>>>>>>>>>>>>Some preprocessing
    prod_map = {"H": 0, "B": 1, "D": 2, "A": 3, "C": 4, "G": 5,
                "F": 6, "I": 7, "E": 8, "J": 9}
    train["product"] = train["product"].map(prod_map)
    data_all["product"] = data_all["product"].map(prod_map)

    #>>>>>>>>>>>>>>>>>>>>>Label encode object types
    lbl = LabelEncoder()
    for col in ["gender", "campaign_id", "webpage_id", "product_category_2"]:
        data_all[col] = lbl.fit_transform(data_all[col].astype(str))
        train[col] = lbl.transform(train[col].astype(str))

    # Sort data for feature calculator
    train.sort_values(by=["user_id", "DateTime"], inplace=True)
    data_all.sort_values(by=["user_id", "DateTime"], inplace=True)
    logger.info(f"Data Loading and preprocessing took {time.time() - t: 6.2f} s")
    t = time.time()


    #>>>>>>>>>>>>>>>>>>>Generate Ad views and count features
    train = get_view_feats(train)
    data_all = get_view_feats(data_all)
    logger.info(f"Time taken for view features {time.time() - t: 6.2f} s")
    t = time.time()

    train = get_count_feats(train)
    data_all = get_count_feats(data_all)
    logger.info(f"Time taken for count features {time.time() - t: 6.2f} s")
    t = time.time()


    #>>>>>>>>>>>>>>>>>>>>>Map features from historical logs
    hist_data_agg = pd.read_csv(str(config.SAVE_PATH / "hist_data_agg.csv"))
    train = pd.merge(train, hist_data_agg, on=["user_id", "product"], how="left")
    data_all = pd.merge(data_all, hist_data_agg, on=["user_id", "product"], how="left")
    del hist_data_agg

    train["hour"] = train["DateTime"].dt.hour
    train["dayofweek"] = train["DateTime"].dt.dayofweek

    data_all["hour"] = data_all["DateTime"].dt.hour
    data_all["dayofweek"] = data_all["DateTime"].dt.dayofweek

    agg4 = pd.read_csv(str(config.SAVE_PATH / "agg4.csv"))
    agg5 = pd.read_csv(str(config.SAVE_PATH / "agg5.csv"))
    train = pd.merge(train, agg4, on=["user_id", "dayofweek"], how="left")
    data_all = pd.merge(data_all, agg4, on=["user_id", "dayofweek"], how="left")

    train["usr_dayofweek_clickratio"] = train["usr_dayofweek_cnt"]/train["total_adclicks"]
    data_all["usr_dayofweek_clickratio"] = data_all["usr_dayofweek_cnt"]/data_all["total_adclicks"]

    train = pd.merge(train, agg5, on=["user_id", "hour"], how="left")
    data_all = pd.merge(data_all, agg5, on=["user_id", "hour"], how="left")

    train["usr_hourofday_clickratio"] = train["usr_hour_cnt"]/train["total_adclicks"]
    data_all["usr_hourofday_clickratio"] = data_all["usr_hour_cnt"]/data_all["total_adclicks"]
    del agg4, agg5

    nmf_feats = pd.read_csv(str(config.SAVE_PATH / "nmf_feats.csv"))
    train = pd.merge(train, nmf_feats, on=["user_id"], how="left")
    data_all = pd.merge(data_all, nmf_feats, on=["user_id"], how="left")
    del nmf_feats

    nmf_day_feats = pd.read_csv(str(config.SAVE_PATH / "nmf_day_feats.csv"))
    train = pd.merge(train, nmf_day_feats, on=["user_id"], how="left")
    data_all = pd.merge(data_all, nmf_day_feats, on=["user_id"], how="left")
    del nmf_day_feats

    user_time_counts = pd.read_csv(str(config.SAVE_PATH / "user_time_counts.csv"))
    train = pd.merge(train, user_time_counts, on=["user_id"], how="left")
    data_all = pd.merge(data_all, user_time_counts, on=["user_id"], how="left")
    cols = user_time_counts.columns
    cols = [col for col in cols if col != "user_id"]
    train[cols].fillna(0)
    data_all[cols].fillna(0)
    del user_time_counts
    logger.info(f"Time taken for loading and mapping historical log features {time.time() - t: 6.2f} s")
    t = time.time()

    train = get_overall_count_feats(train)
    data_all = get_overall_count_feats(data_all)
    logger.info(f"Time taken for overall count {time.time() - t: 6.2f} s")
    t = time.time()

    train = get_extra_feats(train)
    data_all = get_extra_feats(data_all)
    logger.info(f"Time taken for extra features 1 {time.time() - t: 6.2f} s")
    t = time.time()

    if config.validation_flag == 1:
        tr = train.loc[(train["DateTime"] < pd.to_datetime("2017/07/07"))].reset_index()
        val = train.loc[train["DateTime"] >= pd.to_datetime("2017/07/07")].reset_index()
    else:
        tr = train.loc[train["DateTime"] < pd.to_datetime("2017/07/06")].reset_index()
        val = train.loc[train["DateTime"] >= pd.to_datetime("2017/07/06")].reset_index()

    y_tr, y_val = tr["is_click"].values, val["is_click"].values

    tr, val = get_click_feats(tr, val)
    logger.info(f"Time taken for click features 1 {time.time() - t: 6.2f} s")
    t = time.time()

    tr, val = get_target_encoding(tr, val, y_tr)
    logger.info(f"Time taken for likelihood encoding features 1 {time.time() - t: 6.2f} s")
    t = time.time()

    tr = get_ratio_feats(tr)
    val = get_ratio_feats(val)
    logger.info(f"Time taken for ratio features 1 {time.time() - t: 6.2f} s")
    t = time.time()

    feats = [#---------Base features ----------#
             'product', 'campaign_id',
             'webpage_id', 'product_category_1',
             #'product_category_2',
             #user_group_id', #'gender', 'age_level', 'user_depth',
             #'city_development_index',
             'var_1',
             #--------view features-----------#
             'prev_view', 'prev_prod_view', 'prev_wp_view',
             'next_view', 'next_prod_view', 'next_wp_view',
             'view_counts', 'prod_view_counts', 'wp_view_counts',
             'prev_next_rat',
             #---------click features----------#
             #'click_counts', 'wp_click_counts', 'prod_click_counts',
             #target mean
             'prod_wp_tmean', 'prod_wp_entropy', 'usr_tmean', 'prod_grp_tmean', 'usr_age_tmean',
             'usr_age_grp_tmean',
             #---------historical log features----------#
               #'usr_prod_interest',
               'usr_prod_adclicks',
               #'total_interest',
               'total_adclicks',
               #'unique_prods',
               #'hour_mean', 'hour_std',
             'dayofweek_mean', 'dayofweek_std',
              #'dayofyear_min',
              #'dayofyear_max',
               'dayofyear_mean', 'dayofyear_std', 'interest_ratio',
               'product_interest_ratio',
              #'unq_prod_interest', 'last_interest_days',
              # 'last_click_day',
             'product_rank',
             #----------total count features------------#
             'all_counts',
             #'prd_counts',
             #'wp_counts',
             'all_usr_prd_counts', 'all_usr_wp_counts', #'all_prd_cnts',
             'all_grp_camp_counts', 'all_usr_prd_wp_counts', 'all_prd_wp_counts',
             'usr_date_counts', 'usr_wp_date_counts', 'usr_date_nunq_prods',
             #------------ratio_feats------------------#
             'click_ratio', 'clicks_done_ratio', 'prev_next_rat', 'wp_click_ratio',
             'prod_adclick_ratio',
             'curr_prod_count_rat',
             #'hist log - prod counts and day of week counts',
             'prod_cnt1', 'prod_cnt2', 'prod_cnt3', 'prod_cnt4', 'prod_cnt5',
             'prod_cnt6', 'prod_cnt7', 'prod_cnt8',
             #--'prod_cnt9', 'prod_cnt10',
             'day_cnt1', 'day_cnt2', 'day_cnt3', 'day_cnt4', 'day_cnt5',
             'day_cnt6', 'day_cnt7',
            #hist log user time instant stats
             'user_time_counts_mean', 'user_time_counts_std',
             #---'user_time_counts_max', 'user_time_counts_median',
             'user_time_counts_skew',
            #hist log - day of week and hour stats
             #----'usr_dayofweek_cnt', 'usr_hour_cnt'
            ] +\
           ['usr_day_curr_counts', 'usr_prd_day_curr_counts', 'prd_wp_day_curr_counts'] +\
            [f"usr_prd_other_views_{i}" for i in range(10)] + [f"usr_wp_other_views_{i}" for i in range(9)]

    #for f in feats:
    #    sns.distplot(tr[f].fillna(0).replace(np.inf, -1))
    #    sns.distplot(val[f].fillna(0).replace(np.inf, -1))
    #    plt.show()
    #tr.to_csv(str(config.SAVE_PATH / "tr.csv"), index=False)
    #val.to_csv(str(config.SAVE_PATH / "val.csv"), index=False)

    lgb_params = {
        "n_estimators": 10000,
        "learning_rate": 0.02,
        "num_leaves": 8,
        "min_child_samples": 100,
        #"max_bin": 127,
        #"bagging_freq":2,
        "subsample": 0.5,
        "colsample_bytree": 0.26,
        "reg_lambda": 21,
        "reg_alpha": 18,
        "seed": 12345786
        #"min_split": 0.2
    }

    X_tr, X_val = tr[feats], val[feats]
    X_tr = X_tr.fillna(-1)
    X_val = X_val.fillna(-1)

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_tr, y_tr), (X_val, y_val)],
              eval_metric='auc',
              early_stopping_rounds=500,
              verbose=50)

    y_preds_lgb = model.predict_proba(X_val)[:, 1]
    logger.info(f"Validation score is {roc_auc_score(y_val, y_preds_lgb): 9.6f}")
    logger.info(f"Model training took with validation took {time.time() - t: 6.2f} s")
    t = time.time()

    #data_all.to_csv(str(config.SAVE_PATH / "data_all.csv"), index=False)
    lgb_params["n_estimators"] = 6000 #Correct num extimators on increased training dta
    _, test_preds1 = get_test_preds(data_all, feats, lgb_params)
    _, test_preds2 = get_test_preds(data_all, feats, lgb_params, flag=2) #Add diversity to model to make it robust
    logger.info(f"Model retraining with all data and first day left out (3 runs each) took {time.time() - t: 6.2f} s")
    t = time.time()

    #np.save("test_preds1_v2.npy", test_preds1)
    #np.save("test_preds2_v2.npy", test_preds2)
    #np.save("y_preds_lgb_v2.npy", y_preds_lgb)


    test_preds= 0.5*test_preds1 + 0.5*test_preds2
    #sns.distplot(y_preds_lgb)
    #sns.distplot(test_preds)

    logger.info("Write out submissions")
    sub = data_all.loc[data_all.flag1 == "test", ["session_id"]]
    sub["is_click"] = test_preds
    sub.to_csv(str(config.SAVE_PATH / "model10.csv"), index=False)

    #Done!
    logger.info(f"OVERALL TIME TAKEN {(time.time() - start_time)/60: 6.2f} minutes")

