from transformers import LogitsProcessor

class BrandComplianceLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, banned_words):
        self.tokenizer = tokenizer
        self.banned_words = set(w.lower() for w in banned_words)

    def __call__(self, input_ids, scores):
        # aquí va la magia — lo vemos ahora
        return scores