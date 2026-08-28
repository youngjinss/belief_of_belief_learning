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

EXPERIMENTS = ("exp5", "exp6", "exp7", "exp8")

# Forward-pass sums are floating point; compare them with a tolerance rather
# than by equality.
FLOAT_TOLERANCE = 1e-3


def run_probe(experiment):
    """Probe one experiment in its own interpreter, isolating module namespaces."""
    cwd = os.path.join(REPO_ROOT, "script", experiment)
    proc = subprocess.run(
        [sys.executable, PROBE],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{experiment}: probe exited {proc.returncode}\n{proc.stderr[-2000:]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{experiment}: probe emitted invalid JSON: {exc}\n{proc.stdout[:500]}")


def golden_path(experiment):
    return os.path.join(GOLDEN_DIR, f"{experiment}.json")


def load_golden(experiment):
    with open(golden_path(experiment)) as handle:
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
        document = run_probe(experiment)
        with open(golden_path(experiment), "w") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
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
        if not os.path.exists(golden_path(experiment)):
            print(f"MISSING {experiment}: no golden recorded")
            failures += 1
            continue
        differences = list(diff(load_golden(experiment), run_probe(experiment)))
        if differences:
            failures += 1
            print(f"FAIL {experiment}: {len(differences)} difference(s)")
            for line in differences[:20]:
                print(f"     {line}")
            if len(differences) > 20:
                print(f"     ... and {len(differences) - 20} more")
        else:
            print(f"OK   {experiment}")
    return failures


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    if command == "record":
        record()
    elif command == "check":
        sys.exit(1 if check() else 0)
    else:
        sys.exit(f"unknown command: {command}")
