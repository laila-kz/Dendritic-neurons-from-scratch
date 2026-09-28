"""Reference notes for Perforated AI + Weights & Biases dendritic optimization.

This module is a **study artifact**, not part of the experiment pipeline. It
records the API shape for the "Dendritic Optimization" hackathon talk
(Perforated AI / Weights & Biases) and is kept as a reference for how the
optimiser and scheduler are rebuilt whenever Perforated AI restructures a
model and adds new dendrites.

It is intentionally **not** runnable as-is: `yourModel`, `validate` and the
tuning variables below are placeholders for a real PyTorch model. Everything
is defined inside functions and nothing executes on import, so the file can be
imported, linted and unit-tested without the optional
``perforatedai``/``torch``/``wandb`` dependencies being installed.

To use it, copy the relevant function into your own project and replace the
placeholders with your model, data loaders and score function.

Optional extras::

    pip install torch wandb perforatedai
"""

from typing import Any, Callable, Dict, Optional

# Placeholders for a real experiment. Replace these with your own values.
#: Validation score to hand to Perforated AI. Maximised or minimised according
#: to ``maximizing_score`` below.
score: float = 0.0
#: Number of epochs per sweep run.
num_epochs: int = 10
#: Learning rate handed to the optimiser.
learning_rate: float = 1e-3
#: Perforated AI needs a checkpoint name; replace with your own.
save_name: str = "dendritic-sweep"
#: Whether the tracked score should be maximised (accuracy) or minimised (loss).
maximizing_score: bool = True
#: Maximum number of dendrites Perforated AI may add.
cap_at_n: int = 5

#: Sweep hyper-parameters. Values are tried once each.
parameters_dict: Dict[str, Dict[str, Any]] = {
    "dropout": {"values": [0.2, 0.3, 0.4, 0.5]},
    "lr": {"values": [1e-3, 3e-4]},
}


def yourModel(*args: Any, **kwargs: Any) -> Any:
    """Placeholder for your own PyTorch model.

    Replace with a real ``nn.Module``. Perforated AI inspects the module to
    decide where dendrites can be inserted, so the model must be a genuine
    ``nn.Module`` rather than a plain function.
    """
    raise NotImplementedError(
        "Replace `yourModel` with your own torch.nn.Module before using "
        "this reference script."
    )


def validate(model: Any, val_loader: Any) -> float:
    """Placeholder for your validation routine.

    Should return the scalar you want to optimise for (for example validation
    accuracy when ``maximizing_score`` is ``True``).
    """
    raise NotImplementedError(
        "Replace `validate` with your own evaluation loop before using this "
        "reference script."
    )


def build_sweep_config() -> Dict[str, Any]:
    """Return the Weights & Biases sweep configuration.

    Kept separate from any W&B call so the config can be inspected and tested
    without ``wandb`` installed.
    """
    goal = "maximize" if maximizing_score else "minimize"
    return {
        "method": "random",
        "metric": {"name": "ValAcc", "goal": goal},
        "parameters": parameters_dict,
    }


def setup_optimizer(model: Any, lr: float, pai_tracker: Any) -> tuple:
    """Create the Adam optimiser + ReduceLROnPlateau scheduler for a run.

    This must be re-run every time Perforated AI reports that it restructured
    the model and added dendrites: the previous optimiser holds references to
    the old parameter tensors, so continuing to step it would update detached
    parameters and training would silently stop making progress.
    """
    import torch

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pai_tracker.set_optimizer_instance(optimizer)
    pai_tracker.set_optimizer(torch.optim.Adam)
    pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)

    optim_args = {"params": model.parameters(), "lr": lr}
    # Plateau scheduling needs the metric direction to match the objective.
    sched_args = {"mode": "max" if maximizing_score else "min", "patience": 5}
    return pai_tracker.setup_optimizer(model, optim_args, sched_args)


def training_epoch(
    model: Any,
    pai_tracker: Any,
    pai_globals: Any,
    optimizer: Any,
    lr: float,
    epochs: int = 1,
    validate_fn: Optional[Callable[[Any, Any], float]] = None,
    train_step: Optional[Callable[[Any], float]] = None,
) -> Dict[str, Any]:
    """Run training epochs, reporting a validation score after each one.

    ``pai_tracker.add_validation_score`` returns a three-tuple:

    * the (possibly restructured) model,
    * ``restructured`` - ``True`` when new dendrites were just added,
    * ``training_complete`` - ``True`` when the dendrite budget is exhausted.

    When ``restructured`` is ``True`` the optimiser and scheduler must be
    rebuilt, which is the main thing this function exists to demonstrate.

    ``train_step`` and ``validate_fn`` default to the module-level
    placeholders, which raise ``NotImplementedError``.
    """
    validate_fn = validate_fn or validate
    train_step = train_step or (lambda _model: 0.0)

    for epoch in range(epochs):
        train_step(model)

        current_score = validate_fn(model, None)
        pai_globals.pai_tracker.set_current_score(current_score)

        model, restructured, training_complete = pai_tracker.add_validation_score(
            current_score, model
        )

        if training_complete:
            break

        if restructured:
            # New dendrites were added: rebuild the optimiser/scheduler so it
            # points at the new parameters.
            optimizer, _ = setup_optimizer(model, lr, pai_tracker)

    return {
        "model": model,
        "optimizer": optimizer,
        "training_complete": training_complete,
    }


def run_sweep(count: int = 50, project: str = "dendritic-sweeps") -> str:
    """Launch a Weights & Biases sweep agent.

    Returns the sweep id. Requires ``wandb`` to be installed and authenticated.
    """
    import wandb

    wandb.login()
    sweep_id = wandb.sweep(build_sweep_config(), project=project)
    wandb.agent(sweep_id, function=_agent_entrypoint, count=count)
    return sweep_id


def _agent_entrypoint(run: Any) -> None:
    """Body of a single sweep run. Wired up by :func:`run_sweep`."""
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA

    config = run.config
    GPA.pai_tracker.set_cap_at_n(cap_at_n)

    name_str = "_".join(f"{key}-{config[key]}" for key in config)
    run.name = name_str

    lr = config.get("lr", learning_rate)
    model = yourModel(dropout=config["dropout"])
    model = UPA.initialize_pai(model, save_name=name_str, maximizing_score=maximizing_score)

    optimizer, _ = setup_optimizer(model, lr, GPA.pai_tracker)

    training_epoch(
        model,
        GPA.pai_tracker,
        GPA,
        optimizer,
        lr,
        epochs=num_epochs,
    )

    run.log({"ValAcc": score, "epoch": num_epochs})


if __name__ == "__main__":
    print(__doc__)
    print("\nThis is a reference script, not a runnable experiment.")
    print("Wire up `yourModel` and `validate`, install the optional extras")
    print("(`pip install torch wandb perforatedai`), then call `run_sweep()`.")
