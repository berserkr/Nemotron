"""Inspect N-fold expansion under truncate_history_thinking={True, False}.

Renders a 3-turn conversation with reasoning_content on each assistant turn
through create_masked_messages, once per chat template, and prints the
resulting sub-examples so the redundancy concern can be confirmed by eye.

Run from repo root:
    python tests/data_prep/test_e2e_chat_template_examples/test.py
"""

import importlib.util
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]

# Load chat_template.py directly to skip nemotron.data_prep.__init__, which
# imports cosmos_xenna (an optional extra not needed for pure-Jinja rendering).
_spec = importlib.util.spec_from_file_location(
    "_chat_template", REPO_ROOT / "src/nemotron/data_prep/core/chat_template.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
create_masked_messages = _module.create_masked_messages

MESSAGES = [
    {"role": "user", "content": "U1"},
    {"role": "assistant", "reasoning_content": "RC1", "content": "A1"},
    {"role": "user", "content": "U2"},
    {"role": "assistant", "reasoning_content": "RC2", "content": "A2"},
    {"role": "user", "content": "U3"},
    {"role": "assistant", "reasoning_content": "RC3", "content": "A3"},
]

TEMPLATES = ["chat_template.jinja", "chat_template_full_thinking.jinja"]


def render_for_template(template_path: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.chat_template = template_path.read_text()

    sub_examples = create_masked_messages(MESSAGES, tokenizer)

    print(f"\n{'=' * 72}\nTEMPLATE: {template_path.name}\n{'=' * 72}")
    for i, (chunks, _) in enumerate(sub_examples):
        rendered = "".join(c["content"] for c in chunks)
        print(f"\n--- sub-example {i + 1} of {len(sub_examples)} ---")
        print(rendered)


if __name__ == "__main__":
    for name in TEMPLATES:
        render_for_template(REPO_ROOT / name)
