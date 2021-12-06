import argparse

from yaml.loader import SafeLoader
import torch
import optuna

from typing import Tuple,Dict,Any
import yaml

from src.model import Model
from src.utils.torch_utils import model_info,check_runtime
from src.dataloader import create_dataloader
import torch.nn as nn

from src.trainer import TorchTrainer, count_model_params
from src.searchmodel import search_model

import logging
import sys

import wandb
import numpy as np
import random

import torchvision
print(torchvision.__version__)

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ["MobileNetV3", "mobilenet_v3_large", "mobilenet_v3_small"]

model_urls = {
    "mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
    "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
}

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_best_trial_with_condition(optuna_study: optuna.study.Study) -> Dict[str, Any]:
    """Get best trial that satisfies the minimum condition(e.g. accuracy > 0.8).
    Args:
        study : Optuna study object to get trial.
    Returns:
        best_trial : Best trial that satisfies condition.
    """
    df = optuna_study.trials_dataframe().rename(
        columns={
            "values_0": "acc_percent",
            "values_1": "params_nums",
            "values_2": "mean_time",
        }
    )
    ## minimum condition : accuracy >= threshold
    threshold = 0.7
    minimum_cond = df.acc_percent >= threshold

    if minimum_cond.any():
        df_min_cond = df.loc[minimum_cond]
        ## get the best trial idx with lowest parameter numbers
        best_idx = df_min_cond.loc[
            df_min_cond.params_nums == df_min_cond.params_nums.min()
        ].acc_percent.idxmax()

        best_trial_ = optuna_study.trials[best_idx]
        print("Best trial which satisfies the condition")
        print(df.loc[best_idx])
    else:
        print("No trials satisfies minimum condition")
        best_trial_ = None

    return best_trial_


def search_hyperparam(trial:optuna.trial.Trial) -> Dict[str,Any]:
    epochs = trial.suggest_int("epochs", low=100, high=150, step=50)
    img_size = trial.suggest_categorical("img_size", [168, 224]) #112
    #epochs=100
    #batch_size=32
    n_select = trial.suggest_int("n_select", low=0, high=2, step=2)
    batch_size = trial.suggest_int("batch_size", low=16, high=32, step=16)
    return {
        "EPOCHS": epochs,
        "IMG_SIZE": img_size,
        "n_select": n_select,
        "BATCH_SIZE": batch_size,
    }

def objective(trial: optuna.trial.Trial, device) -> Tuple[float, int, float]:
    ##### hyperparams ######
    hyperparams = search_hyperparam(trial) #hyperparam 경우의수 정의해놓은 함수. trial 넘겨줘야됨.
    wandb.log({
        'hyperparams/trial':trial.number,
        'hyperparams/EPOCHS':hyperparams['EPOCHS'],
        'hyperparams/IMG_SIZE':hyperparams['IMG_SIZE'],
        'hyperparams/n_select':hyperparams['n_select'],
        'hyperparams/BATCH_SIZE':hyperparams['BATCH_SIZE'],
        })

    ###### dataset #######
    with open('./configs/data/taco.yaml') as f:
        data_config = yaml.load(f,yaml.SafeLoader)
    #data_config["AUG_TRAIN"] = "randaugment_train"
    data_config["AUG_TRAIN"] = "simple_augment_train"
    #data_config["AUG_TRAIN_PARAMS"] = {
    #    "n_select": hyperparams["n_select"],
    #}
    data_config["AUG_TEST_PARAMS"] = None
    data_config["BATCH_SIZE"] = hyperparams["BATCH_SIZE"]
    data_config["IMG_SIZE"] = hyperparams["IMG_SIZE"]
    wandb.log({
        'hyperparams/VAL_RATIO':data_config["VAL_RATIO"]
    })

    train_loader,val_loader,test_loader = create_dataloader(data_config)

    ###### model #######
    #model_config: Dict[str, Any] = {}
    #mobilenet_small_torch
    with open('./configs/model/mobilenet_small_torch.yaml') as f:
        model_config = yaml.load(f,yaml.SafeLoader)
    #Search space 정의 / suggest_categorical 사용
    #model_config["input_channel"] = 3
    im_size = trial.suggest_categorical("im_size", [32, 64, 128]) #img_size=32
    model_config["INPUT_SIZE"] = [im_size, im_size]
    # model_config["depth_multiple"] = trial.suggest_categorical(
    #     "depth_multiple", [0.25, 0.5, 0.75, 1.0]
    # )
    # model_config["width_multiple"] = trial.suggest_categorical(
    #     "width_multiple", [0.25, 0.5, 0.75, 1.0]
    # )
    #yaml에서의 backbone 부분을 search_model 에서 만듦. #model 경우의수 정의해놓은 함수. trial 넘겨줘야됨.
    #model_config["backbone"], module_info = search_model(trial) 
    model_instance = Model(model_config,verbose=True)
    state_dict = load_state_dict_from_url(model_urls['mobilenet_v3_small'], progress=False)
    new_model_dict = model_instance.model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in new_model_dict}
    new_model_dict.update(state_dict)

    model_instance.model.load_state_dict(new_model_dict)
    model_instance.model.to(device)
    #model.model.to(device) #??
    model_info(model_instance.model)
    mean_time = check_runtime( #모델에서 이미지 1장 처리하는데 걸리는 시간
        model_instance.model,
        [model_config["input_channel"]] + model_config["INPUT_SIZE"],
        device,
    )
    #mean_time = 0.0
    wandb.log({
        'model_config/trial':trial.number,
        'model_config/input_channel':model_config["input_channel"],
        #'model_config/INPUT_SIZE':model_config["INPUT_SIZE"][0],
        'model_config/depth_multiple':model_config["depth_multiple"],
        'model_config/width_multiple':model_config["width_multiple"],
        'model_config/mean_time':mean_time,
    })

    #save data info into exp/latest/data.yml
    with open('./exp/latest4/data.yml','w') as f:
        yaml.dump(data_config,f,default_flow_style=False)
    #save module info into exp/latest/model.yml
    #with open('./exp/latest/model.yml','w') as f:
    #    yaml.dump(model_config,f,default_flow_style=False)
    
    ##### train ######
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_instance.model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        steps_per_epoch=len(train_loader),
        epochs=hyperparams["EPOCHS"],
        pct_start=0.05,
    )
    trainer = TorchTrainer(
        model=model_instance.model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        model_path='./exp/latest4/best.pt', #TODO:args로 받기
        scaler=None, #??
        device=device,
        verbose=1
    )
    trainer.train(train_loader,hyperparams["EPOCHS"],val_dataloader=val_loader,data_config=data_config,model_config=model_config,trial=trial)

    #####test######
    loss, f1_score, acc_percent = trainer.test(model_instance.model, test_dataloader=val_loader,trial=trial.number)
    params_nums = count_model_params(model_instance.model)
    wandb.log({
        'test/trial':trial.number,
        'test/loss':loss,
        'test/f1_score':f1_score,
        'test/acc_percent':acc_percent
    })
    model_info(model_instance.model, verbose=True)
    return f1_score, params_nums, mean_time


#model parsing main function
def main(gpu_id,storage:str=None):
    device = torch.device(f"cuda:{gpu_id}") if (0 <= gpu_id < torch.cuda.device_count()) and (torch.cuda.is_available()) else torch.device("cpu")
    rdb_storage = optuna.storages.RDBStorage(url=storage) if storage is not None else None
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    sampler = optuna.samplers.MOTPESampler() #TPE sampler의 multiobjective 버전

    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"], #왜 이렇게 3개?
        study_name="automl2",
        sampler=sampler,
        storage=rdb_storage,
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial,device), n_trials=15) #objective 함수를 n_trials번 만큼 시도

    #시도 후 결과 분석
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trials:")
    best_trials = study.best_trials

    ## trials that satisfies Pareto Fronts
    for tr in best_trials:
        print(f"  value1:{tr.values[0]}, value2:{tr.values[1]}")
        for key, value in tr.params.items():
            print(f"    {key}:{value}")

    best_trial = get_best_trial_with_condition(study)
    with open('./exp/latest4/config.txt','w') as f:
        f.write(best_trial)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tuning model with Optuna")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("--storage", default="postgresql://optuna:optuna@118.67.132.41:6013/optuna", type=str, help="Optuna database storage path.")
    parser.add_argument("--exp_name",default="exp1")
    parser.add_argument("--seed",default=21)
    args = parser.parse_args()

    wandb.init(project='optimization',name=args.exp_name, entity="cv_14")
    #seed_everything(args.seed)

    main(args.gpu,storage=args.storage if args.storage!="" else None)
    # postgresql://postgres:sohee@127.0.0.1:5432/Optuna

#how to check current study info on postgre
#study = optuna.create_study(study_name='automl1',storage="postgresql://optuna:optuna@118.67.132.41:6013/optuna",load_if_exists=True,directions=["maximize", "minimize", "minimize"])
#study.trials
#study.trials[0]
#study.best_trials