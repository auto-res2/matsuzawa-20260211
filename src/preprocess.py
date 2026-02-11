"""
Data loading and preprocessing for GSM8K dataset.
"""

import re
from typing import List, Tuple, Dict
from datasets import load_dataset


FINAL_RE = re.compile(r"####\s*(-?\d+)")


def extract_final_int(text: str) -> int | None:
    """
    Extract the final integer answer from GSM8K format.
    
    First tries the #### <number> pattern, then falls back to last integer in text.
    
    Args:
        text: Text containing the answer
        
    Returns:
        Integer answer or None if not found
    """
    m = FINAL_RE.search(text)
    if m:
        return int(m.group(1))
    
    # Fallback: last integer in text
    ints = re.findall(r"-?\d+", text)
    return int(ints[-1]) if ints else None


def load_gsm8k(split: str = "train", cache_dir: str = ".cache") -> List[Dict[str, str]]:
    """
    Load GSM8K dataset.
    
    Args:
        split: Dataset split ("train" or "test")
        cache_dir: Cache directory for HuggingFace datasets
        
    Returns:
        List of examples with "question" and "answer" fields
    """
    ds = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    return [{"question": ex["question"], "answer": ex["answer"]} for ex in ds]


def get_gsm8k_pool(n_pool: int, seed: int = 0, cache_dir: str = ".cache") -> List[str]:
    """
    Get a pool of training questions for demonstration selection.
    
    Args:
        n_pool: Number of questions to sample
        seed: Random seed for reproducibility
        cache_dir: Cache directory
        
    Returns:
        List of question strings
    """
    ds = load_dataset("gsm8k", "main", split="train", cache_dir=cache_dir)
    
    if n_pool >= len(ds):
        return [ex["question"] for ex in ds]
    
    # Deterministic sampling
    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(n_pool))
    
    return [ex["question"] for ex in ds]


def get_gsm8k_test(n_test: int = None, cache_dir: str = ".cache") -> List[Tuple[str, int]]:
    """
    Get test questions with gold answers.
    
    Args:
        n_test: Number of test examples (None for all)
        cache_dir: Cache directory
        
    Returns:
        List of (question, gold_answer) tuples
    """
    ds = load_dataset("gsm8k", "main", split="test", cache_dir=cache_dir)
    
    if n_test is not None and n_test < len(ds):
        ds = ds.select(range(n_test))
    
    examples = []
    for ex in ds:
        question = ex["question"]
        gold = extract_final_int(ex["answer"])
        if gold is not None:
            examples.append((question, gold))
    
    return examples


def count_reasoning_steps(text: str) -> int:
    """
    Heuristic to count reasoning steps in a CoT response.
    
    Counts sentences that contain numbers or mathematical operations.
    
    Args:
        text: Chain-of-thought reasoning text
        
    Returns:
        Estimated number of reasoning steps
    """
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    
    step_count = 0
    for sent in sentences:
        # Consider it a reasoning step if it contains numbers or math operators
        if re.search(r"\d+|[\+\-\*/=]", sent):
            step_count += 1
    
    return max(1, step_count)


if __name__ == "__main__":
    # Quick sanity check
    print("Loading GSM8K...")
    
    pool = get_gsm8k_pool(10, seed=0)
    print(f"\nPool sample ({len(pool)} questions):")
    print(pool[0][:100] + "...")
    
    test = get_gsm8k_test(5)
    print(f"\nTest sample ({len(test)} questions):")
    q, a = test[0]
    print(f"Q: {q[:100]}...")
    print(f"A: {a}")
    
    print("\n✓ Preprocessing module OK")
