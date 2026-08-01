class DynamicCompetitorMasking(LogitsProcessor):
    def __init__(self, tokenizer, competitor_tokens, trigger_phrase):
        self.tokenizer = tokenizer
        self.competitor_ids = competitor_tokens  # lista de IDs ya calculada
        self.trigger = trigger_phrase            # p.ej. "presenting our new"

    def __call__(self, input_ids, scores):
        prefix = self.tokenizer.decode(input_ids[0])
        if self.trigger not in prefix.lower():
            return scores  # regla no aplica → no tocamos nada
        for tok_id in self.competitor_ids:
            scores[:, tok_id] = float("-inf")
        return scores