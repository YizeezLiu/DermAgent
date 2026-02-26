"""
GPT-5.2 Clean Baseline — identical pipeline to GPT-4o, only model_id differs.

This is a control experiment to isolate model capability from prompt engineering.
The standard GPT52Model inherits from GPT5Model which adds system prompts,
pre-warm dialogue, prompt overrides, and retry logic for content filtering.
This clean variant strips all of that away for a fair apple-to-apple comparison.
"""

from .gpt4o import GPT4oModel


class GPT52CleanModel(GPT4oModel):
    """GPT-5.2 with GPT-4o's exact prompt pipeline (no anti-filter modifications)."""
    pass
