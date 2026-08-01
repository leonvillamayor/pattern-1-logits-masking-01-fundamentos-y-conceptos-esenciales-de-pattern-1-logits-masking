class SEOComplianceProcessor(LogitsProcessor):
    def __init__(self, tokenizer, positive, banned, beam_size=5):
        self.tokenizer = tokenizer
        self.positive = positive
        self.banned = banned
        self.beam_size = beam_size

    def __call__(self, input_ids, scores):
        # 1) Generamos beam_size continuaciones alternativas muestreando
        #    temporalmente del propio scores antes de aplicar la poda dura.
        continuations = sample_top_k(scores, k=self.beam_size)
        kept = []
        for cont_ids in continuations:
            text = self.tokenizer.decode(cont_ids).lower()
            if any(b in text for b in self.banned):
                continue
            kept.append(cont_ids)
        # 2) Zero-out de todo lo que no haya sobrevivido al filtro.
        mask = torch.full_like(scores, float("-inf"))
        for cont_ids in kept:
            mask[0, cont_ids] = 0.0
        return scores + mask