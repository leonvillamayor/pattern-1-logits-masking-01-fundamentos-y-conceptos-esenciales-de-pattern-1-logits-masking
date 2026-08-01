"""
PATTERN 1: LOGITS MASKING — Episodio 1
5 ejemplos canónicos paso a paso
====================================

Requisitos:
    pip install "transformers>=4.40" torch
    (El Ejemplo 0 usa logits sintéticos; los Ejemplos 4-5 cargan gpt2 ~500MB)

Idea base:
    Un LogitsProcessor modifica los *scores* (logits) ANTES del softmax
    en cada paso de decodificación. Si ponemos float('-inf') en una posición,
    tras el softmax esa probabilidad queda en 0 y el token NUNCA se elige.
"""

from __future__ import annotations

import math
from typing import List

import torch
from transformers import LogitsProcessor, AutoTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# EJEMPLO 1 — ForbiddenWordsLogitsProcessor
# Bloquea una lista de palabras (p. ej. términos ofensivos o marcas prohibidas)
# ──────────────────────────────────────────────────────────────────────────────
class ForbiddenWordsLogitsProcessor(LogitsProcessor):
    """Pone -inf en los ids de las palabras prohibidas."""

    def __init__(self, forbidden_ids: List[List[int]], eos_token_id: int):
        # forbidden_ids: lista de secuencias de ids (una por palabra).
        # Solo bloqueamos el PRIMER token de cada secuencia para no romper
        # la generación a mitad de palabra; las extensiones se bloquean con
        # restricciones de n-gramas (las verás en otro episodio).
        self.first_token_ids: List[int] = [
            seq[0] for seq in forbidden_ids if seq
        ]
        self.eos_token_id = eos_token_id
        # Salvaguarda: jamás bloquees EOS, si no el modelo nunca termina.
        self.first_token_ids = [
            tid for tid in self.first_token_ids if tid != eos_token_id
        ]

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self.first_token_ids:
            scores = scores.clone()              # in-place rompe el grafo
            scores[:, self.first_token_ids] = float("-inf")
        return scores


# ──────────────────────────────────────────────────────────────────────────────
# EJEMPLO 2 — WhitelistLogitsProcessor
# Lo inverso: SOLO se permite generar tokens de una lista cerrada
# ──────────────────────────────────────────────────────────────────────────────
class WhitelistLogitsProcessor(LogitsProcessor):
    """Permite únicamente los ids dados (p. ej. dígitos + un separador)."""

    def __init__(self, allowed_token_ids: List[int]):
        self.allowed_token_ids = allowed_token_ids

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed_token_ids] = 0.0
        # Truco: partimos de -inf y "abrimos" lo permitido. Equivalente a
        # poner -inf en el resto, pero más rápido con vocabularios grandes.
        return scores + mask


# ──────────────────────────────────────────────────────────────────────────────
# EJEMPLO 3 — JsonStartLogitsProcessor
# Fuerza que la primera posición sea SIEMPRE '{' (formato JSON)
# ──────────────────────────────────────────────────────────────────────────────
class JsonStartLogitsProcessor(LogitsProcessor):
    """En el primer paso de generación, solo permite '{' o '['."""

    def __init__(self, open_brace_id: int, open_bracket_id: int):
        self.open_brace_id = open_brace_id
        self.open_bracket_id = open_bracket_id

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # Si el prompt es de longitud L, el siguiente token es el (L+1)-ésimo.
        # Solo actuamos cuando NO hemos generado todavía (longitud == prompt).
        prompt_len = input_ids.shape[1]
        # Trabajamos por cada fila del batch por si hay varios prompts.
        scores = scores.clone()
        for batch_idx in range(scores.shape[0]):
            # Heurística: "todavía no generamos" == sólo hay prompt.
            # Para casos más complejos compara con input_ids[:, prompt_len-1].
            already_generated = input_ids.shape[1]  # override si guardas prompt
            if already_generated == prompt_len:
                scores[batch_idx, :] = float("-inf")
                scores[batch_idx, self.open_brace_id] = 0.0
                scores[batch_idx, self.open_bracket_id] = 0.0
        return scores


# ──────────────────────────────────────────────────────────────────────────────
# EJEMPLO 4 — MinLengthLogitsProcessor
# Prohíbe generar EOS hasta alcanzar una longitud mínima
# ──────────────────────────────────────────────────────────────────────────────
class MinLengthLogitsProcessor(LogitsProcessor):
    """Mientras no se haya generado `min_length` tokens, EOS está vetado."""

    def __init__(self, min_length: int, eos_token_id: int):
        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        cur_len = input_ids.shape[1]
        if cur_len < self.min_length:
            scores = scores.clone()
            scores[:, self.eos_token_id] = float("-inf")
        return scores


# ──────────────────────────────────────────────────────────────────────────────
# EJEMPLO 5 — Composición: varios processors a la vez en model.generate()
# ──────────────────────────────────────────────────────────────────────────────
def demo_composicion() -> None:
    """
    Demostración autocontenida con logits SINTÉTICOS (no descarga modelo).
    Así puedes ejecutar este archivo en CPU sin GPU ni pesos.
    """
    # Vocabulario toy: 5 tokens inventados.
    vocab_size = 5
    tokenizer = AutoTokenizer.from_pretrained(
        "distilgpt2",  # tokenizer barato (~1MB), solo lo usamos como decodificador
    )

    # --- Ejemplo 1 en acción ---------------------------------------------------
    forbidden = [[tokenizer.encode("bad", add_special_tokens=False)]]
    proc1 = ForbiddenWordsLogitsProcessor(forbidden, tokenizer.eos_token_id)

    # --- Ejemplo 2 en acción ---------------------------------------------------
    allow = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(3)]
    proc2 = WhitelistLogitsProcessor(allow)

    # --- Ejemplo 3 en acción ---------------------------------------------------
    proc3 = JsonStartLogitsProcessor(
        open_brace_id=tokenizer.encode("{", add_special_tokens=False)[0],
        open_bracket_id=tokenizer.encode("[", add_special_tokens=False)[0],
    )

    # --- Ejemplo 4 en acción ---------------------------------------------------
    proc4 = MinLengthLogitsProcessor(min_length=4, eos_token_id=tokenizer.eos_token_id)

    # Logits sintéticos:偏爱 token 3 si nada lo enmascara.
    logits = torch.tensor([[1.0, 2.0, 3.0, 100.0, 4.0]])
    print("Logits originales      :", logits)
    print("Token argmax original  :", tokenizer.decode([logits.argmax().item()]))

    # Aplicamos la cadena en el mismo orden que `generate()` los aplicaría.
    chain = [proc1, proc2, proc3, proc4]
    masked = logits
    for proc in chain:
        masked = proc(input_ids=torch.zeros((1, 1), dtype=torch.long), scores=masked)

    print("Logits tras enmascarar :", masked)
    print("Token argmax final     :", tokenizer.decode([masked.argmax().item()]))
    # Observarás que el argmax cambia porque el token favorito fue silenciado
    # por la whitelist y por el bloqueo de '{'/'['.

    # --- Ejemplo 5 (versión REAL con un modelo pequeño) -------------------------
    # Si tienes GPU/descargas, descomenta:
    #
    # from transformers import AutoModelForCausalLM
    # model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    # prompt = tokenizer("Hola,", return_tensors="pt").input_ids
    # out = model.generate(
    #     prompt,
    #     max_new_tokens=10,
    #     logits_processor=[proc1, proc2, proc4],   # composición real
    #     do_sample=False,
    # )
    # print("Generado:", tokenizer.decode(out[0]))


if __name__ == "__main__":
    demo_composicion()