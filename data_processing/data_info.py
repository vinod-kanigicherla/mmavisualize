import pandas as pd 
import pickle 
import datetime

#### DATA RELATED ####

FIGHTERINFO_PATH = "data/fighterinfo.pkl"
FIGHTERINFO = pickle.load(open(FIGHTERINFO_PATH, "rb"))
UNIQUE_FIGHTERS = list(FIGHTERINFO.columns.get_level_values("fighter").unique().values)
FIGHTER_DROPDOWN_OPTIONS = [{"label": f, "value": f} for f in UNIQUE_FIGHTERS]
DATES = FIGHTERINFO.index

STATS = FIGHTERINFO.columns.get_level_values("features").unique().values
STATS_NAMES = ["Wins",
 "Losses",
 "Current Win Streak",
 "Current Loss Streak",
 "Significant Strikes Landed per Fight",
 "Significant Strikes Attempted per Fight",
 "Significant Strikes Accuracy Percentage", 
 "Significant Strikes Absorbed per Fight", 
 "Significant Strikes Defended Percentage", 
 "Takedowns Landed per Fight",
 "Takedowns Attempted per Fight",
 "Takedowns Accuracy Percentage", 
 "Takedowns Absorbed per Fight", 
 "Takedowns Defended Percentage", 
 "Head Strikes Landed per Fight",
 "Head Strikes Attempted per Fight",
 "Head Strikes Accuracy Percentage", 
 "Head Strikes Absorbed per Fight", 
 "Head Strikes Defended Percentage", 
 "Body Strikes Landed per Fight",
 "Body Strikes Attempted per Fight",
 "Body Strikes Accuracy Percentage", 
 "Body Strikes Absorbed per Fight", 
 "Body Strikes Defended Percentage", 
 "Leg Strikes Landed per Fight",
 "Leg Strikes Attempted per Fight",
 "Leg Strikes Accuracy Percentage", 
 "Leg Strikes Absorbed per Fight", 
 "Leg Strikes Defended Percentage", 
 "Distance Strikes Landed per Fight", 
 "Distance Strikes Accuracy Percentage", 
 "Ground Strikes Landed per Fight", 
 "Ground Strikes Accuracy Percentage", 
 "Clinch Strikes Landed per Fight", 
 "Clinch Strikes Accuracy Percentage", 
 "Submissions Attempted per Fight", 
 "Control Time in Seconds per Fight",
 "Reversals per Fight" 
]

STATS_DICT = {sn: s for sn, s in zip(STATS_NAMES, STATS)} 
STATS_DICT_REVERSE = {STATS_DICT[sn]: sn for sn in STATS_DICT.keys()}


def get_min_date(fighter):
    return FIGHTERINFO.index[(FIGHTERINFO.loc[:, fighter].isnull().sum(axis=1) < len(STATS))][0]

def get_cmp_df(f1, f2, date1, date2, stats):
    cmp_df = pd.DataFrame(columns=[f1, "Stat", f2])
    cmp_df["Stat"] = [STATS_DICT_REVERSE[s] for s in stats]

    date1 = DATES[DATES.to_series().apply(lambda x: x < pd.to_datetime(date1))][-1]
    date2 = DATES[DATES.to_series().apply(lambda x: x < pd.to_datetime(date2))][-1]
    
    assert date1 in DATES and date2 in DATES

    cmp_df[f1] = FIGHTERINFO.loc[date1, f1][stats].values
    cmp_df[f2] = FIGHTERINFO.loc[date2, f2][stats].values
    return cmp_df.round(1)


def get_at_ranking_df(stat, date):
    date = DATES[DATES.to_series().apply(lambda x: x < pd.to_datetime(date))][-1]
    top10 = FIGHTERINFO.loc[date, :].xs(stat, level="features").sort_values(ascending=False)[:10]
    top10 = top10.rename(STATS_DICT_REVERSE[stat])
    return pd.DataFrame({"Fighter": top10.index.to_list(), STATS_DICT_REVERSE[stat]: top10.to_list()})


