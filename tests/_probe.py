"""Golden-output probe.

Run as a subprocess with cwd set to a single experiment directory, so that the
flat module namespaces of exp5..exp8 (`config`, `tomnet`, ...) cannot collide in
`sys.modules`. Emits one JSON document on stdout describing the experiment's
observable surface: its resolved configuration and its instantiated model.

Usage:  cd script/exp8 && python ../../tests/_probe.py
"""

import hashlib
import inspect
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, REPO_ROOT)

SEED = 12345


def _reseed(seed):
    """Reset every RNG the experiment code draws from."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def _jsonable(value):
    """Reduce an arbitrary config attribute to something stable and comparable."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    return f"<{type(value).__name__}>"


def probe_config():
    from config import Config

    cfg = Config()
    attrs = {
        name: _jsonable(getattr(cfg, name))
        for name in sorted(vars(cfg))
        if not name.startswith("_")
    }

    # Zero-argument getters are part of the config contract that Phase 3 must
    # preserve when each experiment collapses into a config plus an overlay.
    getters = {}
    for name in sorted(dir(cfg)):
        if not name.startswith("get_") and not name.startswith("is_"):
            continue
        method = getattr(cfg, name)
        if not callable(method):
            continue
        params = [
            p
            for p in inspect.signature(method).parameters.values()
            if p.default is inspect.Parameter.empty
        ]
        if params:
            continue
        # Some getters are deliberately stochastic: get_costs and
        # get_goal_rewards draw from np.random without seeding. Reseed before
        # every call so each recorded value is independent of sweep order.
        _reseed(cfg.seed)
        try:
            getters[name] = _jsonable(method())
        except Exception as exc:
            getters[name] = f"<raises {type(exc).__name__}>"

    return cfg, {"attributes": attrs, "getters": getters}


def probe_model(cfg):
    import numpy as np
    import torch

    from tomnet import ToMnet

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(False)

    kwargs = cfg.get_model_kwargs()
    accepted = set(inspect.signature(ToMnet.__init__).parameters)
    # exp5/exp6 and exp7/exp8 expose different constructor surfaces; pass only
    # what this experiment's ToMnet actually accepts.
    used = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = sorted(set(kwargs) - set(used))

    # Shrink to a size that runs fast while still exercising every layer.
    used["batch"] = 2
    used["time_step"] = 4
    used["max_n_past"] = 2

    model = ToMnet(**used)
    model.eval()

    shapes = {name: list(p.shape) for name, p in sorted(model.named_parameters())}
    n_params = sum(p.numel() for p in model.parameters())

    result = {
        "constructor_params": sorted(accepted - {"self"}),
        "kwargs_used": _jsonable(used),
        "kwargs_dropped": dropped,
        "n_parameters": n_params,
        "n_tensors": len(shapes),
        "param_shapes_sha256": hashlib.sha256(
            json.dumps(shapes, sort_keys=True).encode()
        ).hexdigest(),
        "forward_params": [
            p for p in inspect.signature(model.forward).parameters if p != "kwargs"
        ],
    }

    result["forward"] = probe_forward(model, used)
    return result


def probe_forward(model, used):
    """Run one deterministic forward pass, adapting to the experiment's signature."""
    import torch

    torch.manual_seed(SEED)

    batch = used["batch"]
    seq = used["time_step"]
    n_past = used["max_n_past"]
    channels = used["channels_in"]
    height = used["env_height"]
    width = used["env_width"]

    def rand(*shape):
        return torch.rand(*shape, generator=torch.Generator().manual_seed(SEED))

    past = rand(batch, n_past, seq, channels, height, width)
    recent = rand(batch, seq, channels, height, width)
    current = rand(batch, channels, height, width)
    actions = torch.zeros(batch, seq, dtype=torch.long)

    params = set(inspect.signature(model.forward).parameters)
    if "recent_trajectory" in params:  # exp5 / exp6
        args = dict(
            past_trajectories=past, recent_trajectory=recent, current_state=current
        )
    else:  # exp7 / exp8
        args = dict(
            past_trajectories=past,
            self_states=recent,
            self_actions=actions,
            current_state=current,
        )
        if "oppo_states" in params:
            args["oppo_states"] = recent
            args["oppo_actions"] = actions

    try:
        with torch.no_grad():
            out = model(**args)
    except Exception as exc:
        return {"status": "raises", "error": f"{type(exc).__name__}: {exc}"}

    if not isinstance(out, dict):
        out = {"output": out}

    summary = {}
    for key in sorted(out):
        value = out[key]
        if value is None:
            summary[key] = None
        elif hasattr(value, "shape"):
            summary[key] = {
                "shape": list(value.shape),
                "sum": round(float(value.float().sum()), 4),
            }
        else:
            summary[key] = f"<{type(value).__name__}>"
    return {"status": "ok", "outputs": summary}


def main():
    doc = {"experiment": os.path.basename(os.getcwd()), "seed": SEED}
    cfg, doc["config"] = probe_config()
    try:
        doc["model"] = probe_model(cfg)
    except Exception as exc:
        doc["model"] = {"status": "raises", "error": f"{type(exc).__name__}: {exc}"}
    json.dump(doc, sys.stdout, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
