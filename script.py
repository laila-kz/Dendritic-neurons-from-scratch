'''It is a Python training script that integrates PerforatedAI
and Weights & Biases to train a PyTorch model with dendritic 
optimization, automatically run hyperparameter sweeps, and reset the optimiz
er and scheduler whenever dendrites are added to the model.
'''

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
import torch
import wandb


# ----------------- MODEL + PAI INIT -----------------
model = yourModel()
model = UPA.initialize_pai(
    model,
    save_name=args.save_name,
    maximizing_score=True  # if u need to max ur score set it to true if u wanna min ur value set it to false
)


# ----------------- INITIALIZE OPTIM / SCHED -----------------
#initialize
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
GPA.pai_tracker.set_optimizer_instance(optimizer)
GPA.pai_tracker.set_optimizer(torch.optim.Adam)
GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)

optimArgs = {"params": model.parameters(), "lr": learning_rate}
schedArgs = {"mode": "max", "patience": 5}
optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)


# ----------------- TRAINING LOOP SNIPPET -----------------
for epoch in range(num_epochs):
    # ... forward, loss, backward, optimizer.step() ...
    # suppose `score` = validation accuracy (or -loss if you minimize)
    #add_valdiation_score
    #returns :
    #         model 
    #         restructured : true if dentrite have been added 
    #         training_complete 
    #resetting optimizer and scheduler states if loaded from checkpoint
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(score, model)

    if training_complete:
        break
    elif restructured:
        optimArgs = {"params": model.parameters(), "lr": learning_rate}
        schedArgs = {"mode": "max", "patience": 5}
        optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

    #logging wandb afetr each epoch
    wandb.log({"ValAcc": score, "epoch": epoch})





# ----------------- SWEEP SETUP -----------------
#setup for training loop
wandb.login()

sweep_config = {
    "method": "random",  # grid / random / bayes
}

maximizing_score = True
metric = {"name": "ValAcc", "goal": "maximize" if maximizing_score else "minimize"}  # set goal to false if u wanna min ur value
sweep_config["metric"] = metric

#parameters , config , and id 
#hyper parameters to be swept
parameters_dict = {
    "dropout": {
        "values": [0.2, 0.3, 0.4, 0.5]
    },  # value to try 
    #.....copy them from github
}

#sweep if and config 
sweep_config["parameters"] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project="my first dentrites projects")


#cap_at_n 
GPA.pc.set_cap_at_n(True)  # n is the max number of perforated ai dentrites u want


# ----------------- MAIN + AGENT -----------------
def main(run):
    config = run.config

    #using the values
    GPA.pc.set_dendrite_update_mode(config.get("dendrite_update_mode", "default"))
    # .... other GPA.pc.* from config if needed

    name_str = "_".join([f"{key}-{config[key]}" for key in config.keys()])
    run.name = name_str

    global model
    model = yourModel(dropout=config.dropout)
    model = UPA.initialize_pai(
        model,
        save_name=name_str,
        maximizing_score=maximizing_score
    )

    # re-create optimizer & scheduler for this run
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", learning_rate))
    GPA.pai_tracker.set_optimizer_instance(optimizer)
    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)

    optimArgs = {"params": model.parameters(), "lr": config.get("lr", learning_rate)}
    schedArgs = {"mode": "max" if maximizing_score else "min", "patience": 5}
    optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

    # -------- training loop inside this run --------
    for epoch in range(num_epochs):
        # ... train ...
        # compute validation score
        score = validate(model, val_loader)  # your function

        model_pai, restructured, training_complete = GPA.pai_tracker.add_validation_score(score, model)
        model = model_pai

        if training_complete:
            break
        elif restructured:
            optimArgs = {"params": model.parameters(), "lr": config.get("lr", learning_rate)}
            schedArgs = {"mode": "max" if maximizing_score else "min", "patience": 5}
            optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

        #logging wandb afetr each epoch
        run.log({"ValAcc": score, "epoch": epoch})


def run():
    try:
        with wandb.init() as run:
            main(run)
    except Exception as e:
        import pdb
        pdb.post_mortem()


if __name__ == "__main__":
    # count how many runs to perform 
    wandb.agent(sweep_id, function=run, count=50)
