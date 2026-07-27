import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_Forecasting
from dataset_forecasting import get_dataloader
from utils import train, evaluate, predict


def get_quantile(samples,q,dim=1):
    return torch.quantile(samples,q,dim=dim).cpu().numpy()


torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base_forecasting.yaml")
parser.add_argument("--datatype", type=str, default="gpp")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["timeemb"]    = min(config["model"].get("timeemb", 64), 16)
config["model"]["featureemb"] = min(config["model"].get("featureemb", 64), 16)



config["model"]["is_unconditional"] = args.unconditional

print(json.dumps(config, indent=4))

datafolder = './data/ED/'



net = 'networks'
# for net in networks: 
for varname in ["gpp","nee", "reco"]: 
    if varname == "all":
        target_dim = 3+136
    else:
        target_dim = 1+136
        # 

    print(f"\n=== Running dataset: {net} {varname}===")
    args.datatype = varname
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = "./save/forecasting_" + net + "_"+ varname  + "_6_10x/"
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)


    STAT_PATH = datafolder + 'data_stats_with_NEE_Ra_RECO.npz'
    DATA_PATH = datafolder + f'pft_dataset_12mean_4types_28years_ESACCI_plusAW_{net}.npz'
    X_PATH    = datafolder + f'res_train4_test8_extract_4types_28years_{net}_with_NEE_Ra_RECO.npz'


    train_loader, valid_loader, test_loader, pre_loader, scaler, mean_scaler = get_dataloader(
        datatype=args.datatype,
        device= args.device,
        batch_size=config["train"]["batch_size"],
        stat_path=STAT_PATH,
        data_path=DATA_PATH,
        data_xpath=X_PATH, varname=varname,
    )

    model = CSDI_Forecasting(config, args.device, target_dim).to(args.device)

    model.varname = varname

    # args.modelfolder = "forecasting_" + net + "_"+ varname  + "_6_10x"

    if args.modelfolder == "":
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
    else:
        model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
    model.target_dim = target_dim

    evaluate(
        model,
        pre_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
    )

    # predict(
    #     model,
    #     pre_loader,
    #     nsample=args.nsample,
    #     scaler=scaler,
    #     mean_scaler=mean_scaler,
    #     foldername=foldername,
    # )

    torch.cuda.empty_cache()



    
