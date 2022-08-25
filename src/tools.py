import numpy as np
import random
import os

import subprocess

SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_submission(submission, name, submit_on_kaggle=False):
    last_id = int(sorted(os.listdir("../output/"), reverse=True)[0].split("_")[0])
    new_name = (
        f"/home/rv/projects/Amex_Default_Prediction/output/" + 
        f"{str(last_id + 1).zfill(3)}_{name}.csv"
    )
    
    submission[["customer_ID", "prediction"]].to_csv(new_name, index = False, header=True)
    
    if submit_on_kaggle:
        print(
            subprocess.check_output(
                f'kaggle competitions submit -c amex-default-prediction -f {new_name} -m "Message"'.split()
            )
        )
        
def submit_amex(test, 
                model, n_cols, task_type, 
                n_folds, n_seeds, n_iter, 
                logloss_score, amex_score, 
                submit_on_kaggle=False):
    save_submission(
        submission=test, 
        name=(
            f"model_{model}_features_{n_cols}_tasktype_{task_type}_" +
            f"folds_{n_folds}_seeds_{n_seeds}_iter_{n_iter}_" +
            f"logloss_{str(logloss_score).replace('0.', '')[:4]}_" + 
            f"amex_{str(amex_score).replace('0.', '')[:4]}" 
        ),
        submit_on_kaggle=submit_on_kaggle
    )