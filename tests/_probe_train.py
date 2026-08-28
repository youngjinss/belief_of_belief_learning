"""Training smoke probe: one epoch of one step.

The config/model probe builds a model and runs a forward pass; the agent probe
rolls policies out. Neither touches the training path, yet Phase 2 moved code
that train.py depends on (`create_model`, `ToMnetLoss`, `print_epoch_metrics`,
`save_training_plots`). This probe closes that gap cheaply: it mirrors the real
step in train.py -- forward, ToMnetLoss over the same twelve arguments,
backward, optimizer step -- and records the loss components before and after.

Deliberately one epoch of one step on synthetic tensors, so it runs in about a
second and stays a regression guard rather than a training run.

    cd script/exp8 && python ../../tests/_probe_train.py
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, REPO_ROOT)

SEED = 7777
BATCH = 2
SEQ = 4
N_PAST = 2


def load_model_classes():
    """exp8's model lives in beliefrl.model; exp5-exp7 keep their own tomnet.py."""
    try:
        from tomnet import ToMnet, ToMnetLoss

        return ToMnet, ToMnetLoss, "flat"
    except ImportError:
        from beliefrl.model import ToMnet, ToMnetLoss

        return ToMnet, ToMnetLoss, "beliefrl.model"


def build(cfg, ToMnet):
    import inspect

    kwargs = cfg.get_model_kwargs()
    accepted = set(inspect.signature(ToMnet.__init__).parameters)
    used = {k: v for k, v in kwargs.items() if k in accepted}
    used["batch"] = BATCH
    used["time_step"] = SEQ
    used["max_n_past"] = N_PAST
    return ToMnet(**used), used


def forward_args(model, used):
    """Build one synthetic batch matching this experiment's forward signature."""
    import inspect

    import torch

    gen = torch.Generator().manual_seed(SEED)

    def rand(*shape):
        return torch.rand(*shape, generator=gen)

    channels = used["channels_in"]
    h, w = used["env_height"], used["env_width"]

    past = rand(BATCH, N_PAST, SEQ, channels, h, w)
    recent = rand(BATCH, SEQ, channels, h, w)
    current = rand(BATCH, channels, h, w)
    actions = torch.zeros(BATCH, SEQ, dtype=torch.long)

    params = set(inspect.signature(model.forward).parameters)
    if "recent_trajectory" in params:  # exp5 / exp6
        return dict(
            past_trajectories=past, recent_trajectory=recent, current_state=current
        )
    args = dict(
        past_trajectories=past,
        self_states=recent,
        self_actions=actions,
        current_state=current,
    )
    if "oppo_states" in params:
        args["oppo_states"] = recent
        args["oppo_actions"] = actions
    return args


def targets(outputs):
    """Synthetic labels shaped from the model's own outputs."""
    import torch

    gen = torch.Generator().manual_seed(SEED + 1)

    def classes(logits):
        return torch.randint(
            0, logits.shape[1], (logits.shape[0],), generator=gen, dtype=torch.long
        )

    sr = outputs["sr_pred"]
    consumption = outputs["consumption_logits"]
    return dict(
        action_targets=classes(outputs["action_logits"]),
        goal_targets=classes(outputs["goal_logits"]),
        agent_targets=classes(outputs["agent_logits"]),
        type_targets=classes(outputs["type_logits"]),
        # consumption_loss is BCEWithLogitsLoss: a multi-label target of the
        # same shape as the logits, not class indices.
        consumption_targets=(
            torch.rand(*consumption.shape, generator=gen) > 0.5
        ).float(),
        sr_targets=torch.rand(*sr.shape, generator=gen),
    )


def one_step(cfg):
    import torch

    ToMnet, ToMnetLoss, source = load_model_classes()

    torch.manual_seed(SEED)
    model, used = build(cfg, ToMnet)
    model.train()

    loss_fn = ToMnetLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    args = forward_args(model, used)
    outputs = model(**args)
    if not isinstance(outputs, dict):
        return {"status": "raises", "error": "forward did not return a dict"}

    tgt = targets(outputs)
    # The same twelve positional arguments train.py passes.
    loss_dict = loss_fn(
        outputs["action_logits"],
        outputs["goal_logits"],
        outputs["agent_logits"],
        outputs["type_logits"],
        outputs["consumption_logits"],
        outputs["sr_pred"],
        tgt["action_targets"],
        tgt["goal_targets"],
        tgt["agent_targets"],
        tgt["type_targets"],
        tgt["consumption_targets"],
        tgt["sr_targets"],
    )

    before = {k: round(float(v), 4) for k, v in loss_dict.items() if hasattr(v, "item")}

    optimizer.zero_grad()
    loss_dict["loss"].backward()

    grad_norm = float(
        torch.norm(
            torch.stack(
                [
                    p.grad.detach().norm()
                    for p in model.parameters()
                    if p.grad is not None
                ]
            )
        )
    )
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)

    optimizer.step()

    # A second forward on the same batch: the loss must move after one step.
    with torch.no_grad():
        outputs2 = model(**args)
        loss_dict2 = loss_fn(
            outputs2["action_logits"],
            outputs2["goal_logits"],
            outputs2["agent_logits"],
            outputs2["type_logits"],
            outputs2["consumption_logits"],
            outputs2["sr_pred"],
            tgt["action_targets"],
            tgt["goal_targets"],
            tgt["agent_targets"],
            tgt["type_targets"],
            tgt["consumption_targets"],
            tgt["sr_targets"],
        )
    after = {k: round(float(v), 4) for k, v in loss_dict2.items() if hasattr(v, "item")}

    return {
        "status": "ok",
        "model_source": source,
        "loss_before": before,
        "loss_after": after,
        "loss_decreased": bool(after["loss"] < before["loss"]),
        "grad_norm": round(grad_norm, 3),
        "params_with_grad": n_with_grad,
        "n_parameters": sum(p.numel() for p in model.parameters()),
    }


def main():
    from config import Config

    doc = {"experiment": os.path.basename(os.getcwd()), "seed": SEED}
    try:
        doc["train"] = one_step(Config())
    except Exception as exc:
        doc["train"] = {"status": "raises", "error": f"{type(exc).__name__}: {exc}"}
    json.dump(doc, sys.stdout, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
