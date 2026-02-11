"""
LLM wrappers and metamorphic testing utilities for prompt-only experiments.
"""

import os
import re
import ast
import json
import random
from typing import List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Safe expression evaluation
ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
    ast.USub, ast.UAdd, ast.Mod, ast.Pow, ast.Load
)


def safe_eval_expr(expr: str) -> float:
    """
    Safely evaluate an arithmetic expression using AST whitelist.
    
    Args:
        expr: Arithmetic expression string (e.g., "2 + 3 * 4")
        
    Returns:
        Evaluated result
        
    Raises:
        ValueError: If expression contains disallowed operations
    """
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"Disallowed node: {type(node).__name__}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants allowed")
    return eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {})


# Metamorphic perturbation functions
NAME_POOL = ["Alex", "Blake", "Casey", "Drew", "Evan", "Finley", "Gray", "Harper"]


def rename_people(text: str, seed: int = 0) -> str:
    """Conservative name replacement using a fixed name pool."""
    rnd = random.Random(seed)
    tokens = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    
    uniq = []
    for t in tokens:
        if t not in uniq:
            uniq.append(t)
    
    pool = NAME_POOL.copy()
    rnd.shuffle(pool)
    mapping = {u: pool[i % len(pool)] for i, u in enumerate(uniq[:4])}
    
    out = text
    for src, dst in mapping.items():
        out = re.sub(rf"\b{re.escape(src)}\b", dst, out)
    return out


def reorder_sentences(text: str) -> str:
    """Reverse order of sentences, keeping last (question) fixed."""
    parts = [p.strip() for p in text.split(".") if p.strip()]
    if len(parts) <= 2:
        return text
    
    ctx, last = parts[:-1], parts[-1]
    ctx = list(reversed(ctx))
    return ". ".join(ctx + [last]) + "."


STEM_RULES = [
    (r"\bHow many\b", "What is the number of"),
    (r"\bWhat is\b", "Compute"),
]


def stem_swap(text: str) -> str:
    """Apply fixed question stem substitutions."""
    out = text
    for pat, rep in STEM_RULES:
        out = re.sub(pat, rep, out)
    return out


def normalize_space(text: str) -> str:
    """Normalize whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def metamorphic_variants(q: str, m: int = 5, seed: int = 0) -> List[str]:
    """
    Generate deterministic meaning-preserving variants.
    
    Args:
        q: Original question
        m: Number of variants to generate
        seed: Random seed for deterministic results
        
    Returns:
        List of variant questions (including original as first)
    """
    variants = [q]
    variants.append(normalize_space(q))
    variants.append(stem_swap(q))
    variants.append(rename_people(q, seed=seed))
    variants.append(reorder_sentences(q))
    
    return [normalize_space(v) for v in variants[:m]]


class LocalLLM:
    """Wrapper for locally-hosted LLM (e.g., Qwen on H200)."""
    
    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "bfloat16"):
        """
        Initialize local LLM.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on
            dtype: Data type (bfloat16, float16, float32)
        """
        self.model_name = model_name
        self.device = device
        
        # Map dtype string to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(dtype, torch.bfloat16)
        
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=".cache",
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=".cache",
            torch_dtype=self.dtype,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print(f"✓ Model loaded on {device}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 250,
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        # Format as chat if model expects it
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted = prompt
        
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated.strip()


def gen_expr_final(llm: LocalLLM, q: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Generate structured {expr, final} JSON response.
    
    Args:
        llm: LLM wrapper
        q: Question
        
    Returns:
        (expr, final) tuple or (None, None) if parsing fails
    """
    prompt = (
        "Solve the math word problem. Output ONLY valid JSON with keys: expr, final.\n"
        "- expr: one arithmetic expression using only numbers and + - * / ( )\n"
        "- final: the final integer answer\n"
        "No extra text.\n\n"
        f"Problem: {q}"
    )
    
    try:
        out = llm.generate(prompt, max_new_tokens=160)
        
        # Extract JSON
        j = out.strip()
        if not j.startswith("{"):
            m = re.search(r"\{.*\}", j, flags=re.S)
            j = m.group(0) if m else "{}"
        
        data = json.loads(j)
        expr = str(data.get("expr", "")).strip()
        final = data.get("final", None)
        
        if final is not None and str(final).strip().lstrip('-').isdigit():
            final = int(final)
        else:
            final = None
        
        return expr, final
    except Exception:
        return None, None


def answer_only(llm: LocalLLM, q: str) -> Optional[int]:
    """
    Get answer-only response (no CoT).
    
    Args:
        llm: LLM wrapper
        q: Question
        
    Returns:
        Integer answer or None
    """
    prompt = (
        "Solve the problem. Output ONLY the final answer as '#### <number>'.\n\n"
        f"Problem: {q}"
    )
    
    try:
        from .preprocess import extract_final_int
        out = llm.generate(prompt, max_new_tokens=80)
        return extract_final_int(out)
    except Exception:
        return None


def invariance_pass_rate(
    llm: LocalLLM,
    q: str,
    final: int,
    m: int = 5,
    seed: int = 0
) -> float:
    """
    Compute metamorphic invariance pass rate.
    
    Args:
        llm: LLM wrapper
        q: Original question
        final: Expected answer
        m: Number of variants
        seed: Random seed
        
    Returns:
        Pass rate (fraction of variants that match final)
    """
    variants = metamorphic_variants(q, m=m, seed=seed)
    
    ok = 0
    for v in variants:
        a = answer_only(llm, v)
        if a is not None and a == final:
            ok += 1
    
    return ok / len(variants)
