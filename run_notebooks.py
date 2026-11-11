"""Execute the five notebooks in dependency order, with saved outputs."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parent
ORDER = ["preprocessing", "logistic_regression", "naive_bayes", "decision_tree", "neural_network"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", choices=ORDER)
    args = parser.parse_args()
    report = {}
    os.chdir(ROOT)
    shell = InteractiveShell.instance()
    for name in args.notebooks or ORDER:
        print(f"Running {name}...", flush=True)
        start = time.monotonic()
        path = ROOT / f"{name}.ipynb"
        notebook = nbformat.read(path, as_version=4)
        shell.reset(new_session=False)
        try:
            count = 0
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                count += 1
                print(f"  cell {count}", flush=True)
                with capture_output() as captured:
                    result = shell.run_cell(cell.source, store_history=False)
                cell.execution_count = count
                cell.outputs = []
                if captured.stdout:
                    cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
                if captured.stderr:
                    cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
                for output in captured.outputs:
                    cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
                error = result.error_before_exec or result.error_in_exec
                if error:
                    cell.outputs.append(nbformat.v4.new_output("error", ename=type(error).__name__, evalue=str(error), traceback=traceback.format_exception(error)))
                    raise error
            report[name] = {"status": "passed", "seconds": round(time.monotonic()-start, 2), "code_cells": sum(c.cell_type=="code" for c in notebook.cells)}
        finally:
            nbformat.write(notebook, path)
            (ROOT / "validation.json").write_text(json.dumps(report, indent=2)+"\n")
        print(f"Passed {name}: {report[name]['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
