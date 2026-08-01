class DocAwareAutocomplete(LogitsProcessor):
    def __init__(self, tokenizer, document_text, top_k=5):
        self.tokenizer = tokenizer
        self.doc = document_text.lower()
        self.top_k = top_k

    def __call__(self, input_ids, scores):
        # Filtramos el vocabulario a tokens cuyo texto aparezca en el documento.
        allowed = []
        for tok_id in self.tokenizer.get_vocab().values():
            piece = self.tokenizer.decode([tok_id]).strip().lower()
            if piece and piece in self.doc:
                allowed.append(tok_id)
        mask = torch.full_like(scores, float("-inf"))
        for tid in allowed:
            mask[:, tid] = 0.0
        return scores + mask