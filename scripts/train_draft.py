"""Backward-compatible entrypoint: redirects to scripts.train_stage2."""
import runpy

if __name__ == "__main__":
    runpy.run_module("scripts.train_stage2", run_name="__main__")
