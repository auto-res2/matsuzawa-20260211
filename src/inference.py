"""
Inference-only demonstration selection and evaluation for prompt-tuning experiments.
This module implements both MPC-AutoCoT and baseline Auto-CoT methods.
"""

import json
import numpy as np
from typing import List, Dict, Tuple
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from .preprocess import extract_final_int, count_reasoning_steps, get_gsm8k_pool, get_gsm8k_test
from .model import LocalLLM, gen_expr_final, answer_only, safe_eval_expr, invariance_pass_rate


def build_demos_mpc(
    llm: LocalLLM,
    questions: List[str],
    k: int = 8,
    seed: int = 0,
    n_candidates_per_cluster: int = 10,
    alpha: float = 0.15,
    m_vars: int = 5,
    token_budget: int = 120,
) -> Tuple[str, Dict]:
    """
    Build demonstrations using MPC-AutoCoT method.
    
    Args:
        llm: LLM wrapper
        questions: Pool of candidate questions
        k: Number of demonstrations to select
        seed: Random seed
        n_candidates_per_cluster: Candidates to evaluate per cluster
        alpha: Token penalty weight
        m_vars: Number of metamorphic variants
        token_budget: Token budget for penalty calculation
        
    Returns:
        (demos_text, stats) tuple
    """
    print(f"Building {k} MPC-AutoCoT demos from pool of {len(questions)}...")
    
    # Diversity clustering
    embedder = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=".cache")
    X = embedder.encode(questions, normalize_embeddings=True, show_progress_bar=False)
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = km.fit_predict(X)
    
    demos = []
    stats = {
        "accepted": 0,
        "json_fail": 0,
        "exec_fail": 0,
        "inv_fail": 0,
        "inv_scores": [],
    }
    
    for c in range(k):
        idxs = [i for i, l in enumerate(labels) if l == c]
        centroid = km.cluster_centers_[c]
        idxs.sort(key=lambda i: float(-np.dot(X[i], centroid)))
        cand_idxs = idxs[:min(len(idxs), n_candidates_per_cluster)]
        
        best_demo, best_score, best_inv = None, -1e9, 0.0
        
        for i in cand_idxs:
            q = questions[i]
            
            try:
                # Generate structured solution
                expr, final = gen_expr_final(llm, q)
                if not expr or final is None:
                    stats["json_fail"] += 1
                    continue
                
                # Execution gate
                val = safe_eval_expr(expr)
                exec_ok = int(round(float(val)) == final)
                if not exec_ok:
                    stats["exec_fail"] += 1
                    continue
                
                # Metamorphic invariance gate
                inv = invariance_pass_rate(llm, q, final, m=m_vars, seed=seed)
                
                if inv < 1.0:
                    stats["inv_fail"] += 1
                    continue
                
                # Scoring
                demo = f"Q: {q}\nA: expr={expr}\n#### {final}\n"
                tok_pen = len(demo.split()) / token_budget
                score = inv - alpha * tok_pen
                
                if score > best_score:
                    best_score, best_demo, best_inv = score, demo, inv
                
                # Early exit if perfect
                if inv == 1.0:
                    break
                    
            except Exception as e:
                stats["exec_fail"] += 1
                continue
        
        if best_demo is None:
            # Fallback: use first candidate with placeholder
            q = questions[cand_idxs[0]]
            best_demo = f"Q: {q}\nA: (no demo found)\n"
            best_inv = 0.0
        else:
            stats["accepted"] += 1
        
        stats["inv_scores"].append(best_inv)
        demos.append(best_demo)
    
    demos_text = "\n".join(demos) + "\n"
    
    print(f"  Accepted: {stats['accepted']}/{k}")
    print(f"  Mean inv: {np.mean(stats['inv_scores']):.3f}")
    
    return demos_text, stats


def build_demos_autocot(
    llm: LocalLLM,
    questions: List[str],
    k: int = 8,
    seed: int = 0,
    n_candidates_per_cluster: int = 10,
    min_steps: int = 2,
    max_steps: int = 8,
    min_tokens: int = 20,
    max_tokens: int = 150,
) -> Tuple[str, Dict]:
    """
    Build demonstrations using baseline Auto-CoT method.
    
    Args:
        llm: LLM wrapper
        questions: Pool of candidate questions
        k: Number of demonstrations to select
        seed: Random seed
        n_candidates_per_cluster: Candidates to evaluate per cluster
        min_steps, max_steps: Step count heuristics
        min_tokens, max_tokens: Token count heuristics
        
    Returns:
        (demos_text, stats) tuple
    """
    print(f"Building {k} Auto-CoT demos from pool of {len(questions)}...")
    
    # Diversity clustering (same as MPC)
    embedder = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=".cache")
    X = embedder.encode(questions, normalize_embeddings=True, show_progress_bar=False)
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = km.fit_predict(X)
    
    demos = []
    stats = {
        "accepted": 0,
        "step_filter": 0,
        "token_filter": 0,
    }
    
    for c in range(k):
        idxs = [i for i, l in enumerate(labels) if l == c]
        centroid = km.cluster_centers_[c]
        idxs.sort(key=lambda i: float(-np.dot(X[i], centroid)))
        cand_idxs = idxs[:min(len(idxs), n_candidates_per_cluster)]
        
        best_demo = None
        
        for i in cand_idxs:
            q = questions[i]
            
            # Generate free-form CoT
            prompt = f"Solve step-by-step:\n{q}\nSolution:"
            try:
                cot = llm.generate(prompt, max_new_tokens=250)
                
                # Heuristic filtering
                steps = count_reasoning_steps(cot)
                tokens = len(cot.split())
                
                if not (min_steps <= steps <= max_steps):
                    stats["step_filter"] += 1
                    continue
                
                if not (min_tokens <= tokens <= max_tokens):
                    stats["token_filter"] += 1
                    continue
                
                # Accept first that passes filters
                best_demo = f"Q: {q}\nA: {cot}\n"
                stats["accepted"] += 1
                break
                
            except Exception:
                continue
        
        if best_demo is None:
            # Fallback
            q = questions[cand_idxs[0]]
            best_demo = f"Q: {q}\nA: (no demo found)\n"
        
        demos.append(best_demo)
    
    demos_text = "\n".join(demos) + "\n"
    
    print(f"  Accepted: {stats['accepted']}/{k}")
    
    return demos_text, stats


def answer_with_demos(llm: LocalLLM, demos_text: str, q: str) -> int | None:
    """
    Answer question using demonstrations in context.
    
    Args:
        llm: LLM wrapper
        demos_text: Demonstration context
        q: Test question
        
    Returns:
        Predicted integer answer or None
    """
    prompt = demos_text + f"Q: {q}\nA:"
    out = llm.generate(prompt, max_new_tokens=260)
    return extract_final_int(out)


def run_inference(cfg: DictConfig, mode: str = "main") -> Dict:
    """
    Run full inference pipeline for one seed.
    
    Args:
        cfg: Hydra config
        mode: "main" or "sanity_check"
        
    Returns:
        Results dictionary
    """
    seed = cfg.inference.seeds[0] if mode == "main" else 0
    
    # Adjust parameters for sanity_check
    if mode == "sanity_check":
        n_pool = 20
        n_test = 10
        k_demos = 3
        n_candidates = 3
        print("SANITY_CHECK mode: reduced scale")
    else:
        n_pool = cfg.method.n_pool
        n_test = cfg.dataset.n_test
        k_demos = cfg.method.k_demos
        n_candidates = cfg.method.n_candidates_per_cluster
    
    # Initialize LLM
    llm = LocalLLM(
        model_name=cfg.model.name,
        device=cfg.model.device,
        dtype=cfg.model.dtype
    )
    
    # Load data
    pool = get_gsm8k_pool(n_pool, seed=seed, cache_dir=cfg.dataset.cache_dir)
    test = get_gsm8k_test(n_test, cache_dir=cfg.dataset.cache_dir)
    
    # Build demonstrations
    if cfg.method.type == "proposed":
        demos_text, demo_stats = build_demos_mpc(
            llm, pool, k=k_demos, seed=seed,
            n_candidates_per_cluster=n_candidates,
            alpha=cfg.method.alpha,
            m_vars=cfg.method.metamorphic_variants,
            token_budget=cfg.method.token_budget,
        )
    else:  # comparative (Auto-CoT)
        demos_text, demo_stats = build_demos_autocot(
            llm, pool, k=k_demos, seed=seed,
            n_candidates_per_cluster=n_candidates,
            min_steps=cfg.method.min_steps,
            max_steps=cfg.method.max_steps,
            min_tokens=cfg.method.min_tokens,
            max_tokens=cfg.method.max_tokens,
        )
    
    # Evaluate on test set
    print(f"Evaluating on {len(test)} test examples...")
    correct = 0
    predictions = []
    
    for q, gold in test:
        pred = answer_with_demos(llm, demos_text, q)
        predictions.append({
            "question": q,
            "gold": gold,
            "pred": pred,
            "correct": pred is not None and pred == gold,
        })
        if pred is not None and pred == gold:
            correct += 1
    
    accuracy = correct / len(test)
    
    results = {
        "seed": seed,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test),
        "demo_stats": demo_stats,
        "demo_context_tokens": len(demos_text.split()),
        "predictions": predictions,
    }
    
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{len(test)})")
    
    return results
