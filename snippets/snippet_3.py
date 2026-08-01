def apply_rules(self, generated_text, scores):
    banned_ids = [self._word_to_token_id(w) for w in self.banned_words]
    for token_id in banned_ids:
        scores[:, token_id] = float('-inf')
    return scores