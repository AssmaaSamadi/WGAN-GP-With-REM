from src.GAN.WGAN.Trainer import objective
from src.Metrics.precision_recall import knn_precision_recall_features
import optuna
import logging
import sys
import pickle

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "OptunaWgan2"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)

study = optuna.create_study(study_name=study_name, direction="maximize",storage=storage_name,load_if_exists=True)
study.optimize(objective, n_trials=40)

best_par = study.best_params
best_trial =study.best_trial

df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete'], axis=1)  # Exclude columns
df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
df = df.drop('state', axis=1)                 # Exclude state column
df = df.sort_values('value')                  # Sort based on accuracy
#save al trials params and results
df.to_csv('results/GAN/optuna_results2.csv', index=False)  # Save to csv file

#save best params to json
with open('results/GAN/best_params2.pkl', 'wb') as f:
    pickle.dump(best_par, f)

# Save best_trial to a JSON file
with open('results/GAN/best_trial2.pkl', 'wb') as f:
    pickle.dump(best_trial, f)
