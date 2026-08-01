pipe = pipeline("text-generation", model="gpt2")
processor = BrandComplianceLogitsProcessor(
    tokenizer=pipe.tokenizer,
    banned_words=["competidor", "rival"]
)
output = pipe(
    "Describe nuestro producto:",
    max_new_tokens=50,
    num_beams=5,
    logits_processor=[processor]
)