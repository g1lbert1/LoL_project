#Random Forest implementation to classify win/loss of League of Legends games (specifically solo/duo queue type)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score

# SETTINGS
JSON_PATHS     = [
    "league_match_data_20250415_123727.json",
    "league_match_data_20250421_232906.json"
]
QUEUE_TYPES    = "ranked_solo_duo_games"  # or None for all queues
USE_CHAMPS     = True
USE_RANKS      = True
USE_DIFFS      = False
RANDOM_SEED    = 42
RANK_STRATEGY  = "impute"                 # "impute" or "drop"
roles          = ["TOP","JUNGLE","MID","BOT","SUPPORT"]

# LOAD & FLATTEN
def load_matches(path, queue_types=None):
    with open(path, 'r') as f:
        data = json.load(f)
    rows = []
    for qt, tiers in data.items():
        if queue_types and qt != queue_types:
            continue
        for tier, tier_dict in tiers.items():
            for division, division_dict in tier_dict.items():
                for match_id, match in division_dict.items():
                    row = {
                        "matchId": match_id,
                        "queueType": qt,
                        "blue_win": int(match["winningTeam"] == 100)
                    }
                    for r in roles:
                        champs = match.get("champions", {}).get(r, {})
                        ranks  = match.get("roles", {}).get(r, {})
                        diff   = match.get("matchup_differentials", {}).get(r)
                        row[f"{r}_champ_blue"] = champs.get("blue")
                        row[f"{r}_champ_red" ] = champs.get("red")
                        row[f"{r}_rank_blue"  ] = ranks .get("blue")
                        row[f"{r}_rank_red"   ] = ranks .get("red")
                        row[f"{r}_diff"       ] = diff
                    rows.append(row)
    return pd.DataFrame(rows)

# LOAD & CONCAT
df = pd.concat([load_matches(p, QUEUE_TYPES) for p in JSON_PATHS],
               ignore_index=True)

# REMOVE DUPLICATES
df = df.drop_duplicates(subset="matchId")

# COERCE RANK COLUMNS TO NUMERIC
rank_cols = [f"{r}_rank_blue" for r in roles] + [f"{r}_rank_red" for r in roles]
df[rank_cols] = df[rank_cols].apply(pd.to_numeric, errors='coerce')

# DROP ROWS WITH MISSING CHAMPIONS
champ_cols = [f"{r}_champ_blue" for r in roles] + [f"{r}_champ_red" for r in roles]
df = df.dropna(subset=champ_cols)

# OPTIONAL: DROP ROWS WITH MISSING DIFFS
if USE_DIFFS:
    diff_cols = [f"{r}_diff" for r in roles]
    df = df.dropna(subset=diff_cols)

# HANDLE NULL RANKS
if RANK_STRATEGY == "drop":
    df = df.dropna(subset=rank_cols)

elif RANK_STRATEGY == "impute":
    mean_ranks = df[rank_cols].mean(axis=1).round(-2).astype(int)
    for col in rank_cols:
        df[col] = df[col].fillna(mean_ranks)
    df[rank_cols] = df[rank_cols].astype(int)

else:
    raise ValueError("RANK_STRATEGY must be 'drop' or 'impute'")

# BUILD FEATURE LISTS
feat_cols = []
cat_cols  = []
num_cols  = []

if USE_CHAMPS:
    champs = [f"{r}_champ_blue" for r in roles] + [f"{r}_champ_red" for r in roles]
    cat_cols += champs
    feat_cols += champs

if USE_RANKS:
    num_cols += rank_cols
    feat_cols += rank_cols

if USE_DIFFS:
    diffs = [f"{r}_diff" for r in roles]
    num_cols += diffs
    feat_cols += diffs

cat_cols += ["queueType"]
feat_cols += ["queueType"]

# PREPARE FEATURES & LABELS
X = df[feat_cols]
y = df["blue_win"].values


#one hot encoding categorical columns
X_encoded = pd.get_dummies(X, columns=cat_cols)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=RANDOM_SEED)


#training the random forest model (n_estimators = 400 due to the amount of data (~ 15k rows total in df))
RF = RandomForestClassifier(n_estimators=400, max_depth=20, min_samples_leaf=10, random_state=RANDOM_SEED)
RF.fit(X_train, y_train)
train_acc = accuracy_score(y_train, RF.predict(X_train))
test_acc = accuracy_score(y_test, RF.predict(X_test))
#testing overfitting
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
#5fold calculation
scores = cross_val_score(RF, X_encoded, y, cv=5)
print(f"5-Fold CV Accuracy: {scores.mean():.4f}")

y_pred = RF.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred, labels=[1,0])
#printing tree (max-depth=3)
print("\nConfusion Matrix:\n", conf_matrix)
plt.figure(figsize=(20,10))
plot_tree(RF.estimators_[0], feature_names=X_train.columns, max_depth=3, filled=True, fontsize=8)
plt.title("Decision Tree 0 (max depth = 3)")
plt.show()


