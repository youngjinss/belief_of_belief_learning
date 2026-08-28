"""Record and compare golden outputs for the experiment pipelines.

The refactor collapses exp5..exp8 onto a shared core. These goldens are the
contract that migration must not break: they capture each experiment's resolved
configuration and its instantiated ToMnet, recorded against the pre-refactor
code so that every later phase is verifiable.

    python tests/golden.py record    # write tests/golden/<exp>.json
    python tests/golden.py check     # compare current code against them
"""

import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "golden")
PROBE = os.path.join(REPO_ROOT, "tests", "_probe.py")
AGENT_PROBE = os.path.join(REPO_ROOT, "tests", "_probe_agents.py")
TRAIN_PROBE = os.path.join(REPO_ROOT, "tests", "_probe_train.py")

# Two probes per experiment: "" is the config/model probe, "agents" rolls every
# agent out in a fixed-seed environment. Agent behaviour is not observable from
# the config/model probe, and Phase 2 changes agent imports, so it needs its own
# contract.
KINDS = ("", "agents", "train")

EXPERIMENTS = ("exp5", "exp6", "exp7", "exp8")

# Forward-pass sums are floating point; compare them with a tolerance rather
# than by equality.
FLOAT_TOLERANCE = 1e-3


def run_probe(experiment, kind=""):
    """Probe one experiment in its own interpreter, isolating module namespaces."""
    cwd = os.path.join(REPO_ROOT, "script", experiment)
    # PYTHONHASHSEED must be pinned in the child's environment, not from inside
    # it: CPython fixes hash randomisation at interpreter startup, so the
    # `os.environ["PYTHONHASHSEED"] = str(seed)` inside utils.set_seed() has no
    # effect. Without this, agents whose behaviour depends on set/dict iteration
    # order produce a different action sequence on every run.
    env = dict(os.environ, PYTHONHASHSEED="0")
    proc = subprocess.run(
        [sys.executable, {"agents": AGENT_PROBE, "train": TRAIN_PROBE}.get(kind, PROBE)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{experiment}: probe exited {proc.returncode}\n{proc.stderr[-2000:]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{experiment}: probe emitted invalid JSON: {exc}\n{proc.stdout[:500]}")


def golden_path(experiment, kind=""):
    suffix = f".{kind}" if kind else ""
    return os.path.join(GOLDEN_DIR, f"{experiment}{suffix}.json")


def load_golden(experiment, kind=""):
    with open(golden_path(experiment, kind)) as handle:
        return json.load(handle)


def diff(expected, actual, path=""):
    """Walk two probe documents and yield a human-readable difference per mismatch."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            here = f"{path}.{key}" if path else key
            if key not in actual:
                yield f"{here}: missing (was {expected[key]!r})"
            elif key not in expected:
                yield f"{here}: unexpected (now {actual[key]!r})"
            else:
                yield from diff(expected[key], actual[key], here)
        return

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            yield f"{path}: length {len(expected)} -> {len(actual)}"
            return
        for index, (want, got) in enumerate(zip(expected, actual)):
            yield from diff(want, got, f"{path}[{index}]")
        return

    if isinstance(expected, float) and isinstance(actual, (int, float)):
        if abs(expected - actual) > FLOAT_TOLERANCE:
            yield f"{path}: {expected} -> {actual}"
        return

    if expected != actual:
        yield f"{path}: {expected!r} -> {actual!r}"


def record():
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    for experiment in EXPERIMENTS:
        for kind in KINDS:
            document = run_probe(experiment, kind)
            with open(golden_path(experiment, kind), "w") as handle:
                json.dump(document, handle, indent=2, sort_keys=True)
                handle.write("\n")
            if kind == "train":
                t = document["train"]
                if t["status"] == "ok":
                    print(
                        f"recorded {experiment}.train: 1 step, "
                        f"loss {t['loss_before']['loss']} -> {t['loss_after']['loss']}, "
                        f"grad_norm {t['grad_norm']} (model from {t['model_source']})"
                    )
                else:
                    print(f"recorded {experiment}.train: {t['status']} {t.get('error','')[:60]}")
            elif kind == "agents":
                agents = document["agents"]
                ok = sum(1 for v in agents.values() if v["status"] == "ok")
                print(
                    f"recorded {experiment}.agents: {len(agents)} rollouts, "
                    f"{ok} ok, {len(agents) - ok} raising "
                    f"(env {document.get('env_version')})"
                )
            else:
                model = document.get("model", {})
                print(
                    f"recorded {experiment}: "
                    f"{len(document['config']['attributes'])} config attrs, "
                    f"{model.get('n_parameters', 0):,} model params, "
                    f"forward={model.get('forward', {}).get('status')}"
                )


def check():
    failures = 0
    for experiment in EXPERIMENTS:
        for kind in KINDS:
            label = f"{experiment}.{kind}" if kind else experiment
            if not os.path.exists(golden_path(experiment, kind)):
                print(f"MISSING {label}: no golden recorded")
                failures += 1
                continue
            differences = list(
                diff(load_golden(experiment, kind), run_probe(experiment, kind))
            )
            if differences:
                failures += 1
                print(f"FAIL {label}: {len(differences)} difference(s)")
                for line in differences[:20]:
                    print(f"     {line}")
                if len(differences) > 20:
                    print(f"     ... and {len(differences) - 20} more")
            else:
                print(f"OK   {label}")
    return failures


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    if command == "record":
        record()
    elif command == "check":
        sys.exit(1 if check() else 0)
    else:
        sys.exit(f"unknown command: {command}")
