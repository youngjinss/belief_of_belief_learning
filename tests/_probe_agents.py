"""Agent behavioural probe.

Phase 2 moves exp8's agents/ package into the beliefrl core, which means
breaking its import-time coupling to exp8's `config` and `utils`. Five agent
modules currently run `config = Config(); set_seed(config.seed)` at import time,
so importing them mutates global RNG state. Changing that could silently alter
every action sequence, and the config/model goldens would not notice.

This probe pins the actual behaviour: it rolls each agent out in a real,
fixed-seed AchieverBlocker environment and records the action sequence it
produces. Run once per experiment directory, like `_probe.py`.

    cd script/exp8 && python ../../tests/_probe_agents.py
"""

import inspect
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "lib"))
sys.path.insert(0, os.path.join(REPO_ROOT, "lib", "env"))

SEED = 4242
STEPS = 30
GRID = "9x9"

# Fixed reward/cost vectors so the environment is fully determined by SEED.
# Both are colour-keyed dicts: the environment does `max(preference,
# key=preference.get)` and generate.py does `goal_rewards.get(color, 0.0)`.
PREFERENCE = {"red": 1.0, "green": 0.5, "blue": 0.25, "yellow": 0.125}
COST = {"red": 0.25, "green": 0.25, "blue": 0.25, "yellow": 0.25}


def env_version():
    """Use whichever environment version this experiment's Config selects.

    exp5/exp6/exp7 target the v1 AchieverBlocker; exp8 switched to v2 for
    partial observability. Probing each against its own environment keeps the
    recorded behaviour faithful to how that experiment actually runs.
    """
    try:
        from config import Config

        return "v2" if "v2" in Config().get_env_name() else "v1"
    except Exception:
        return "v1"


def build_env(observability):
    if env_version() == "v2":
        from gym_minigrid.envs.achiever_blocker_v2 import AchieverBlocker9x9EnvV2

        return AchieverBlocker9x9EnvV2(
            preference=PREFERENCE,
            cost=COST,
            max_steps=100,
            observability=observability,
            partial_view_size=6,
        )

    from gym_minigrid.envs.achiever_blocker import AchieverBlocker9x9Env

    return AchieverBlocker9x9Env(preference=PREFERENCE, cost=COST, max_steps=100)


def load_agents():
    """Import every agent class this experiment exposes.

    exp5/6/7 keep flat achievers.py / blockers.py modules; exp8 uses the
    agents/ package. Try both so one probe covers every experiment.
    """
    found = {}

    flat = [
        ("achievers", ["AStarAgent", "RandomAgent", "Level0ValueAchiever",
                       "Level1ValueAchiever"]),
        ("blockers", ["RandomAgent", "GoalDirectAgent", "RandomlySelectedAgent",
                      "RuleBasedAgent", "Level0ValueBlocker", "Level1ValueBlocker"]),
    ]
    for module_name, class_names in flat:
        try:
            module = __import__(module_name)
        except Exception:
            continue
        for class_name in class_names:
            obj = getattr(module, class_name, None)
            if obj is not None:
                found[f"{module_name}.{class_name}"] = obj

    package = [
        ("beliefrl.agents.achiever.astar", "AStarAgent"),
        ("beliefrl.agents.achiever.random", "RandomAgent"),
        ("beliefrl.agents.achiever.level0value", "Level0ValueAchiever"),
        ("beliefrl.agents.achiever.level1value", "Level1ValueAchiever"),
        ("beliefrl.agents.blocker.random", "RandomAgent"),
        ("beliefrl.agents.blocker.goaldirected", "GoalDirectAgent"),
        ("beliefrl.agents.blocker.randomlyselect", "RandomlySelectedAgent"),
        ("beliefrl.agents.blocker.rulebased", "RuleBasedAgent"),
        ("beliefrl.agents.blocker.level0value", "Level0ValueBlocker"),
        ("beliefrl.agents.blocker.level1value", "Level1ValueBlocker"),
    ]
    for module_path, class_name in package:
        try:
            module = __import__(module_path, fromlist=[class_name])
        except Exception:
            continue
        obj = getattr(module, class_name, None)
        if obj is not None:
            found[f"{module_path}.{class_name}"] = obj

    return found


def construct(agent_class, env, observability):
    """Instantiate an agent, supplying only the arguments it declares."""
    accepted = set(inspect.signature(agent_class.__init__).parameters)
    kwargs = {}
    if "observability" in accepted:
        kwargs["observability"] = observability
    if "action_space" in accepted:
        kwargs["action_space"] = env.action_space
    if "grid_width" in accepted:
        kwargs["grid_width"] = 9
    if "grid_height" in accepted:
        kwargs["grid_height"] = 9
    if "goal_rewards" in accepted:
        kwargs["goal_rewards"] = PREFERENCE
    return agent_class(**kwargs)


def rollout(agent_class, observability):
    """Record the action sequence one agent produces from a fixed seed."""
    import random

    import numpy as np

    random.seed(SEED)
    np.random.seed(SEED)

    env = build_env(observability)
    env.seed(SEED)
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    agent = construct(agent_class, env, observability)
    if hasattr(agent, "reset"):
        agent.reset()

    actions = []
    for _ in range(STEPS):
        if hasattr(agent, "update_observation"):
            try:
                agent.update_observation(obs)
            except Exception as exc:
                return {"status": "raises", "at": "update_observation",
                        "error": f"{type(exc).__name__}: {exc}"}
        try:
            action = agent.get_action(obs)
        except Exception as exc:
            return {"status": "raises", "at": "get_action",
                    "error": f"{type(exc).__name__}: {exc}"}
        actions.append(int(action) if action is not None else None)

        # Agents consume the whole observation dict: it carries both agents'
        # views plus shared state (achiever_pos, blocker_pos, key_positions,
        # door_positions, wall_positions, grid_info). Narrowing it to one
        # agent's sub-dict breaks every blocker and every Level1 agent.
        step_result = env.step({"achiever": action, "blocker": 0})
        obs = step_result[0]
        done = step_result[2] if len(step_result) > 2 else False
        if done:
            break

    return {"status": "ok", "n_actions": len(actions), "actions": actions}


def main():
    doc = {
        "experiment": os.path.basename(os.getcwd()),
        "seed": SEED,
        "steps": STEPS,
        "grid": GRID,
        "agents": {},
    }
    # The v1 environment has no observability switch; only v2 does.
    modes = ("full", "partial") if env_version() == "v2" else ("full",)
    doc["env_version"] = env_version()
    for name, agent_class in sorted(load_agents().items()):
        for observability in modes:
            key = f"{name}[{observability}]"
            try:
                doc["agents"][key] = rollout(agent_class, observability)
            except Exception as exc:
                doc["agents"][key] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
    json.dump(doc, sys.stdout, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
