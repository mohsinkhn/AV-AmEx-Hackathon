import time
import pandas as pd
import numpy as np

import config


#Author - Mohsin

if __name__ == "__main__":
    start_time = time.time()
    #Read data
    hist_data = pd.read_csv(str(config.HISTORICAL_LOGS_PATH), parse_dates=["DateTime"])

    #DateTime features
    hist_data["hour"] = hist_data["DateTime"].dt.hour
    hist_data["dayofweek"] = hist_data["DateTime"].dt.dayofweek
    hist_data["dayofyear"] = hist_data["DateTime"].dt.dayofyear

    #Map action and product columns to integers - for faster calculations
    action_map = {"view": 0, "interest": 1}
    hist_data.action = hist_data.action.map(action_map)

    prod_map = {"H": 0, "B": 1, "D": 2, "A": 3, "C": 4, "G": 5,
                "F": 6, "I": 7, "E": 8, "J": 9}
    hist_data["product"] = hist_data["product"].map(prod_map)

    #aggregation on historical data by user_id
    print("Getting aggregation on user_id")
    usr_data = hist_data.groupby("user_id").agg({"action": ["sum", "count"], "product": "nunique",
                                                "hour": ["mean", "std"], "dayofweek":["mean", "std"],
                                                "dayofyear": ["min", "max", "mean", "std"]})
    usr_data.columns = ["total_interest", "total_adclicks", "unique_prods", "hour_mean", "hour_std",
                       "dayofweek_mean", "dayofweek_std", "dayofyear_min", "dayofyear_max", "dayofyear_mean", "dayofyear_std"]
    usr_data = usr_data.reset_index(drop=False)


    # aggregation on historical data by user_id and product
    print("Getting aggregation on user_id and product")
    usr_prod_data = hist_data.groupby(["user_id", "product"]).agg({"action": ["sum", "count"]})
    usr_prod_data.columns = ["usr_prod_interest", "usr_prod_adclicks"]
    usr_prod_data = usr_prod_data.reset_index(drop=False)

    #merge both data
    all_data = pd.merge(usr_prod_data, usr_data, on=["user_id"], how="left")
    del usr_data, usr_prod_data


    #Some ratio's
    all_data["interest_ratio"] = all_data["total_interest"]/all_data["total_adclicks"]
    all_data["product_interest_ratio"] = all_data["usr_prod_interest"] / all_data["usr_prod_adclicks"]

    #More features -
    ##  unique product in which interest shown,
    ##  last day when either ad was viewed
    ##  last day interest shown
    ##  Rank ofproduct for user based on ad views
    print("Getting ratio and rank features")
    all_data["unq_prod_interest"] = all_data.user_id.map(hist_data.loc[hist_data["action"] == 1].groupby("user_id")["product"].nunique()).fillna(0)
    all_data["last_interest_days"]  = all_data.user_id.map(hist_data.loc[hist_data["action"] == 1].groupby("user_id")["dayofyear"].apply(lambda x: 182 - x.max())).fillna(0)
    all_data["last_click_day"]  = all_data.user_id.map(hist_data.groupby("user_id")["dayofyear"].apply(lambda x: 182 - x.max())).fillna(0)
    all_data["product_rank"] = all_data.groupby("user_id")["usr_prod_adclicks"].rank()
    all_data.to_csv(str(config.SAVE_PATH / "hist_data_agg.csv"), index=False)
    del all_data

    #datetime aggregation features
    ## How many adviews on day of week
    ## How many adviews on each  hour of day
    print("Getting aggregation on user_id by dayofweek and hour")
    agg4 = hist_data.groupby(["user_id", "dayofweek"]).size()
    agg4.name = "usr_dayofweek_cnt"
    agg4 = agg4.reset_index(drop=False)
    agg4.to_csv(str(config.SAVE_PATH / "agg4.csv"), index=False)
    del agg4

    agg5 = hist_data.groupby(["user_id", "hour"]).size()
    agg5.name = "usr_hour_cnt"
    agg5 = agg5.reset_index(drop=False)
    agg5.to_csv(str(config.SAVE_PATH / "agg5.csv"), index=False)
    del agg5


    #We also want to map product views on all products to user.
    #This can be thought of as a information for clustering similar users together
    #It is possible to apply some decomposition technique like NMF and reduce dimensionn further.
    print("Getting aggregation on all products and dayofweek")
    tmp = hist_data.groupby(["user_id", "product"]).size().unstack().fillna(0)
    nmf_feats = pd.DataFrame(np.log1p(tmp.values), columns=[f"prod_cnt{i+1}" for i in range(10)])
    nmf_feats["user_id"] = tmp.index.values
    nmf_feats.to_csv(str(config.SAVE_PATH / "nmf_feats.csv"), index=False)
    del tmp, nmf_feats

    #Similar to above we can use day of week activity as a way of clustering users together
    tmp = hist_data.groupby(["user_id", "dayofweek"]).size().unstack().fillna(0)
    nmf_feats = pd.DataFrame(np.log1p(tmp.values), columns=[f"day_cnt{i+1}" for i in range(7)])
    nmf_feats["user_id"] = tmp.index.values
    nmf_feats.to_csv(str(config.SAVE_PATH / "nmf_day_feats.csv"), index=False)
    del tmp, nmf_feats


    #There were many instances where for a user at a given datetime multiple ads are shown
    #Here we get statistics on historically how many ads are shown to a user
    print("Getting statistics on multiple ads to same user")
    user_time = hist_data.groupby(["user_id", "DateTime"]).size()
    user_time.name = "user_time_counts"
    user_time = user_time.reset_index()
    user_time = user_time.groupby("user_id").agg({"user_time_counts": ["mean", "std", "max", "median", "skew"]})
    user_time.columns = ["user_time_counts_mean", "user_time_counts_std", "user_time_counts_max",
                         "user_time_counts_median", "user_time_counts_skew"]
    user_time = user_time.reset_index()
    user_time.to_csv(str(config.SAVE_PATH / "user_time_counts.csv"))

    del user_time, hist_data

    #Done!!
    t = time.time()
    print(f"Total time taken for getting features {t - start_time:6.2f} s")

