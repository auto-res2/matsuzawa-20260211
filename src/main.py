"""
Main entry point for prompt-only experiments.
Orchestrates inference runs with Hydra configuration.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from .inference import run_inference


def apply_mode_overrides(cfg: DictConfig, mode: str) -> DictConfig:
    """
    Apply mode-specific configuration overrides.
    
    Args:
        cfg: Base configuration
        mode: "main", "sanity_check", or "pilot"
        
    Returns:
        Modified configuration
    """
    if mode == "sanity_check":
        # Minimal execution for validation
        OmegaConf.set_struct(cfg, False)
        cfg.wandb.mode = "disabled"
        cfg.dataset.n_test = 10
        cfg.method.n_pool = 20
        cfg.method.k_demos = 3
        cfg.method.n_candidates_per_cluster = 3
        OmegaConf.set_struct(cfg, True)
    
    return cfg


def sanity_validation(results: dict, mode: str) -> bool:
    """
    Perform sanity validation on results.
    
    Args:
        results: Results dictionary from run_inference
        mode: Execution mode
        
    Returns:
        True if validation passes
    """
    if mode != "sanity_check":
        return True
    
    # Extract metrics
    accuracy = results.get("accuracy", 0)
    total = results.get("total", 0)
    correct = results.get("correct", 0)
    
    # Validation criteria
    reasons = []
    
    # Check that some samples were processed
    if total < 5:
        reasons.append("too_few_samples")
    
    # Check that accuracy is not 0 (at least some correct)
    if total > 0 and correct == 0:
        reasons.append("zero_accuracy")
    
    # Check for valid accuracy
    if not (0 <= accuracy <= 1):
        reasons.append("invalid_accuracy")
    
    # Print summary
    summary = {
        "samples": total,
        "correct": correct,
        "accuracy": accuracy,
    }
    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")
    
    # Print verdict
    if reasons:
        print(f"SANITY_VALIDATION: FAIL reason={','.join(reasons)}")
        return False
    else:
        print("SANITY_VALIDATION: PASS")
        return True


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main orchestration function.
    
    Args:
        cfg: Hydra configuration
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", action="store_true", help="Run in main mode")
    parser.add_argument("--sanity_check", action="store_true", help="Run in sanity_check mode")
    parser.add_argument("--pilot", action="store_true", help="Run in pilot mode")
    args, unknown = parser.parse_known_args()
    
    # Determine mode
    if args.sanity_check:
        mode = "sanity_check"
    elif args.pilot:
        mode = "pilot"
    else:
        mode = "main"
    
    print(f"=== Running {cfg.run.run_id} in {mode} mode ===\n")
    
    # Apply mode overrides
    cfg = apply_mode_overrides(cfg, mode)
    
    # Create results directory
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize WandB if enabled
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.wandb.mode,
        )
        print(f"WandB run: {wandb.run.url}\n")
    
    try:
        # Run inference
        results = run_inference(cfg, mode=mode)
        
        # Save results
        results_file = results_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_file}")
        
        # Log to WandB
        if cfg.wandb.mode != "disabled":
            wandb.log({
                "accuracy": results["accuracy"],
                "correct": results["correct"],
                "total": results["total"],
                "demo_context_tokens": results["demo_context_tokens"],
            })
            
            # Save metrics to summary
            for key in ["accuracy", "correct", "total", "demo_context_tokens"]:
                wandb.summary[key] = results[key]
        
        # Sanity validation
        sanity_validation(results, mode)
        
    finally:
        if cfg.wandb.mode != "disabled":
            wandb.finish()


if __name__ == "__main__":
    main()
