"""
Episodio 1 - Anatomía técnica de LogitsProcessor en Hugging Face Transformers
Patrón: Logits Masking - Fundamentos
"""

from __future__ import annotations

from typing import List

import torch
from transformers import LogitsProcessor


class AllowedTokensLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor que restringe la generación a un conjunto de tokens permitidos.

    Anatomía:
    - __init__: recibe configuración inmutable (la lista de IDs permitidos).
    - __call__: invocado en CADA paso de decodificación con (input_ids, scores).
                 Devuelve los logits modificados listos para softmax/sampling.
    """

    def __init__(self, allowed_token_ids: List[int]):
        # Guardamos la máscara como buffer de PyTorch en GPU si está disponible.
        # `scores` vendrá en el mismo device, así que moveremos en __call__.
        self.allowed_token_ids = allowed_token_ids
        if not allowed_token_ids:
            raise ValueError("La lista de tokens permitidos no puede estar vacía.")

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            input_ids: (batch_size, sequence_length) — historial generado.
            scores:    (batch_size, vocab_size)      — logits del siguiente token.

        Returns:
            Logits con -inf en todas las posiciones excepto los tokens permitidos.
        """
        # 1. Construimos una máscara booleana del tamaño del vocabulario.
        vocab_size = scores.shape[-1]

        # Truco eficiente: partimos de "todo prohibido" (-inf) y desbloitamos
        # solo los IDs permitidos poniéndolos a 0.
        mask = torch.full(
            (vocab_size,),
            fill_value=float("-inf"),
            dtype=scores.dtype,
            device=scores.device,
        )
        mask[self.allowed_token_ids] = 0.0

        # 2. Aplicamos la máscara sobre todos los elementos del batch.
        #    broadcast: (vocab_size,) -> (batch_size, vocab_size)
        masked_scores = scores + mask

        return masked_scores


# ---------------------------------------------------------------------------
# Demo autocontenida (sin descargar modelo): simulamos la llamada que
# haría el `model.generate()` internamente en cada paso de decodificación.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Vocabulario "de juguete" de 50 tokens; permitimos solo [3, 7, 11].
    VOCAB_SIZE = 50
    BATCH_SIZE = 2

    processor = AllowedTokensLogitsProcessor(allowed_token_ids=[3, 7, 11])

    # Simulamos un batch de logits "crudos" (como los saldría la cabeza LM).
    fake_scores = torch.randn(BATCH_SIZE, VOCAB_SIZE)
    fake_input_ids = torch.zeros((BATCH_SIZE, 4), dtype=torch.long)

    # Esto es EXACTAMENTE lo que Transformers hace internamente en cada paso.
    new_scores = processor(input_ids=fake_input_ids, scores=fake_scores)

    # Verificación: fuera de los IDs permitidos, todo debe ser -inf.
    assert torch.isinf(new_scores).all(dim=-1).sum() == BATCH_SIZE * (VOCAB_SIZE - 3)
    # Y los IDs permitidos deben conservar su valor original.
    for tid in [3, 7, 11]:
        assert torch.allclose(new_scores[:, tid], fake_scores[:, tid])

    print("✅ Máscara aplicada correctamente.")
    print(f"   - Vocab size:      {VOCAB_SIZE}")
    print(f"   - Tokens permitidos: {[3, 7, 11]}")
    print(f"   - Posiciones enmascaradas a -inf: {VOCAB_SIZE - 3} por batch")