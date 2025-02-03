import argparse
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import datetime
import time
import matplotlib.pyplot as plt
from torchinfo import summary
import yaml
import json
import sys
import copy

sys.path.append("..")
from lib.utils import (
    MaskedMAELoss,
    print_log,
    seed_everything,
    set_cpu_num,
    CustomJSONEncoder,
)
from lib.metrics import RMSE, MAE, MAPE
from lib.data_prepare import get_dataloaders_from_index_data, get_k_fold_dataloaders
from model.GRU_Transformer import GRU_Transformer

# ! X shape: (B, T, N, C)


@torch.no_grad()
def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

    return np.mean(batch_loss_list)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    y = []
    out = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()

    return y, out


def train_one_epoch(
    model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=None
):
    global cfg, global_iter_count, global_target_length

    model.train()
    batch_loss_list = []
    for x_batch, y_batch in trainset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    epoch_loss = np.mean(batch_loss_list)
    scheduler.step()

    return epoch_loss


def train(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    scheduler,
    criterion,
    clip_grad=0,
    max_epochs=100,  
    early_stop=10,
    verbose=1,
    plot=False,
    log=None,
    save=None,
):
    model = model.to(DEVICE)

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=log
        )
        train_loss_list.append(train_loss)

        val_loss = eval_model(model, valset_loader, criterion)
        val_loss_list.append(val_loss)

        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                " \tTrain Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss,
                log=log,
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= early_stop:
                break

    model.load_state_dict(best_state_dict)
    y_true, y_pred = predict(model, trainset_loader)
    train_rmse = RMSE(y_true, y_pred)
    train_mae = MAE(y_true, y_pred)
    train_mape = MAPE(y_true, y_pred)

    y_true, y_pred = predict(model, valset_loader)
    val_rmse = RMSE(y_true, y_pred)
    val_mae = MAE(y_true, y_pred)
    val_mape = MAPE(y_true, y_pred)

    out_str = f"Early stopping at epoch: {epoch+1}\n"
    out_str += f"Best at epoch {best_epoch+1}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )
    print_log(out_str, log=log)

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    if save:
        torch.save(best_state_dict, save)
    return model


@torch.no_grad()
def test_model(model, testset_loader, log=None):
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()
    y_true, y_pred = predict(model, testset_loader)
    end = time.time()

    rmse_all = RMSE(y_true, y_pred)
    mae_all = MAE(y_true, y_pred)
    mape_all = MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse = RMSE(y_true[:, i, :], y_pred[:, i, :])
        mae = MAE(y_true[:, i, :], y_pred[:, i, :])
        mape = MAPE(y_true[:, i, :], y_pred[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )

    print_log(out_str, log=log, end="")
    print_log("Inference time: %.2f s" % (end - start), log=log)


def train_k_fold_models(
    model_class,
    model_args, 
    fold_dataloaders,
    criterion,
    optimizer_class,
    optimizer_args,
    scheduler_class, 
    scheduler_args,
    device,
    testset_loader,
    max_epochs=100,
    early_stop=10,
    log=None
):
    models = []
    val_scores = []
    test_metrics = []
    

    save_dir = "../saved_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for fold, (train_loader, val_loader, scaler) in enumerate(fold_dataloaders):
        print_log(f"\nTraining fold {fold+1}", log=log)
        
        model = model_class(**model_args).to(device)
        optimizer = optimizer_class(model.parameters(), **optimizer_args)
        scheduler = scheduler_class(optimizer, **scheduler_args)
        
        save_path = os.path.join(save_dir, f"model_fold{fold+1}_{timestamp}.pt")
        
        model = train(
            model, train_loader, val_loader,
            optimizer, scheduler, criterion,
            max_epochs=max_epochs,
            early_stop=early_stop,
            log=log,
            save=save_path  
        )
        
        val_loss = eval_model(model, val_loader, criterion)
        val_scores.append(val_loss)
        models.append(model)
        
        print_log(f"Fold {fold+1} validation loss: {val_loss:.5f}", log=log)
        print_log(f"Model saved to {save_path}", log=log)
        
        # Test results
        print_log(f"\nFold {fold+1} Test Results:", log=log)
        y_true, y_pred = predict(model, testset_loader)
        fold_rmse = RMSE(y_true, y_pred)
        fold_mae = MAE(y_true, y_pred)
        fold_mape = MAPE(y_true, y_pred)
        
        test_metrics.append({
            'rmse': fold_rmse,
            'mae': fold_mae,
            'mape': fold_mape
        })
        
        print_log(f"Fold {fold+1} Test RMSE: {fold_rmse:.5f}", log=log)
        print_log(f"Fold {fold+1} Test MAE: {fold_mae:.5f}", log=log)
        print_log(f"Fold {fold+1} Test MAPE: {fold_mape:.5f}", log=log)
    
    ensemble_results = {
        'val_scores': val_scores,
        'test_metrics': test_metrics,
        'timestamp': timestamp
    }
    results_path = os.path.join(save_dir, f"ensemble_results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(ensemble_results, f, cls=CustomJSONEncoder)
    print_log(f"\nEnsemble results saved to {results_path}", log=log)
    
    avg_rmse = np.mean([m['rmse'] for m in test_metrics])
    avg_mae = np.mean([m['mae'] for m in test_metrics])
    avg_mape = np.mean([m['mape'] for m in test_metrics])
    
    print_log("\nAverage Test Results across all folds:", log=log)
    print_log(f"Avg RMSE: {avg_rmse:.5f} ± {np.std([m['rmse'] for m in test_metrics]):.5f}", log=log)
    print_log(f"Avg MAE: {avg_mae:.5f} ± {np.std([m['mae'] for m in test_metrics]):.5f}", log=log)
    print_log(f"Avg MAPE: {avg_mape:.5f} ± {np.std([m['mape'] for m in test_metrics]):.5f}", log=log)
    
    return models, val_scores, test_metrics


if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="pems08")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    parser.add_argument("-m", "--mask_type", type=str, default="none")  
    args = parser.parse_args()

    seed = torch.randint(1000, (1,)) # set random seed here
    seed_everything(seed)
    set_cpu_num(1)

    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    dataset = dataset.upper()
    mask_type = args.mask_type  
    data_path = f"../data/{dataset}"
    model_name = GRU_Transformer.__name__

    with open(f"{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    # -------------------------------- load model -------------------------------- #

    model = GRU_Transformer(**cfg["model_args"], mask_type=mask_type)  # Pass mask_type to model

    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"../logs/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}-{dataset}-{mask_type}-{now}.log")  # include mask_type in log name
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    # ------------------------------- load dataset ------------------------------- #

    print_log(dataset, log=log)
    (
        trainset_loader,
        valset_loader,
        testset_loader,
        SCALER,
    ) = get_dataloaders_from_index_data(
        data_path,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg.get("batch_size", 64),
        log=log,
    )
    print_log(log=log)

    # --------------------------- set model saving path -------------------------- #

    save_path = f"../saved_models/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{dataset}-{mask_type}-{now}.pt")  

    # ---------------------- set loss, optimizer, scheduler ---------------------- #

    if dataset in ("METRLA", "PEMSBAY"):
        criterion = MaskedMAELoss()
    elif dataset in ("PEMS03", "PEMS04", "PEMS07", "PEMS08"):
        criterion = nn.HuberLoss()
    else:
        raise ValueError("Unsupported dataset.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["milestones"],
        gamma=cfg.get("lr_decay_rate", 0.1),
        verbose=False,
    )

    # --------------------------- print model structure -------------------------- #

    print_log("---------", model_name, "---------", log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )
    print_log(
        summary(
            model,
            [
                cfg["batch_size"],
                cfg["in_steps"],
                cfg["num_nodes"],
                next(iter(trainset_loader))[0].shape[-1],
            ],
            verbose=0,  # avoid print twice
        ),
        log=log,
    )
    print_log(log=log)

    # --------------------------- train and test model --------------------------- #

    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)

    # Get k-fold dataloaders
    fold_dataloaders = get_k_fold_dataloaders(
        data_path,
        k_folds=5,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg.get("batch_size", 64),
        log=log
    )
    
    # Train k-fold models
    models, val_scores, test_metrics = train_k_fold_models(
        model_class=GRU_Transformer,
        model_args=cfg["model_args"],
        fold_dataloaders=fold_dataloaders,
        criterion=criterion,
        optimizer_class=torch.optim.Adam,
        optimizer_args={
            "lr": cfg["lr"],
            "weight_decay": cfg.get("weight_decay", 0),
            "eps": cfg.get("eps", 1e-8)
        },
        scheduler_class=torch.optim.lr_scheduler.MultiStepLR,
        scheduler_args={
            "milestones": cfg["milestones"],
            "gamma": cfg.get("lr_decay_rate", 0.1)
        },
        device=DEVICE,
        testset_loader=testset_loader,
        max_epochs=cfg.get("max_epochs", 200),
        early_stop=cfg.get("early_stop", 10),
        log=log
    )

    log.close()