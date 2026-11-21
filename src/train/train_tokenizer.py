"""Build a character-level tokenizer and save it (locally by default, or push to the HF Hub).

This builds a deterministic, training-free WordLevel tokenizer that maps each
character in `VOCAB_CHARS` to its own token. This pattern is useful for small
from-scratch models on synthetic tasks where the input alphabet is small and fixed.

TODO (you must implement this for your task):
  1. `VOCAB_CHARS`  -- every character that can appear in a prompt or answer.
  2. `SAMPLE_TEXTS` -- a few representative strings for the round-trip sanity
     checks at the bottom of this script.

The tokenizer is saved locally to --output-dir, then LOADED BACK the way training and
inference load it (AutoTokenizer.from_pretrained), and every check below runs on that loaded
copy -- what you test is exactly what the model will use. With --hub-name it is then also pushed
to the HF Hub, but only if every check passed.

--add-bos makes the tokenizer put a <bos> token at the start of every sequence it encodes.
Training and inference both follow the tokenizer (src/model.py), so they always agree. A fixed
first token gives attention heads a "resting place" to attend to when nothing else is relevant,
which many models use; leave it off for the simplest possible sequences.

See the commented-out example below for a small self-contained illustration
(a single-digit arithmetic task).
"""

import argparse

from tokenizers import Regex, Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

# --------------------------------------------------------------------------- #
# TODO: fill these two lists in for your task.
#
#   VOCAB_CHARS:  EVERY character that can appear in any prompt or answer your model
#                 will ever see -- digits, operators, letters, spaces, newline, etc.
#                 The tokenizer gives each one its own token id. Anything you forget to
#                 list becomes the <unk> token, so be exhaustive. Use sorted(...) so the
#                 vocabulary order is deterministic across machines.
#
#   SAMPLE_TEXTS: a handful of representative strings (using only VOCAB_CHARS). The block
#                 at the bottom of this file encodes then decodes each one and checks it
#                 round-trips exactly -- a quick sanity check that your vocab is complete.
# --------------------------------------------------------------------------- #
#
# Example (a single-digit addition task with prompts like "7+5="):
#
#     VOCAB_CHARS = sorted("0123456789+=\n")
#     SAMPLE_TEXTS = [
#         "2+3=5\n",
#         "7+5=12\n",
#         "1+1=2\n4+0=4\n2+6=8\n6+4=10\n5+6=11\n8+1=9\n9+8=17\n3+3=",
#     ]
#
VOCAB_CHARS: list[str] = []
SAMPLE_TEXTS: list[str] = []

# --------------------------------------------------------------------------- #

_SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
_VOCAB = {tok: i for i, tok in enumerate(_SPECIAL_TOKENS + VOCAB_CHARS)}


def build_tokenizer(add_bos: bool = False) -> PreTrainedTokenizerFast:
    """Build a deterministic character-level tokenizer over VOCAB_CHARS.

    A tokenizer maps text <-> integer token IDs. Here we use the simplest possible scheme:
    one token per character (a WordLevel vocab plus a Split pre-tokenizer that isolates every
    single character). This is ideal for small synthetic tasks because it's transparent and has
    a tiny vocabulary -- no subword merges to reason about. The Fuse decoder just glues the
    characters back together when decoding.

    Args:
        add_bos: If True, a post-processor puts <bos> at the start of every encoded sequence.
    """
    tok_obj = Tokenizer(WordLevel(vocab=_VOCAB, unk_token="<unk>"))
    tok_obj.pre_tokenizer = Split(pattern=Regex(r"[\s\S]"), behavior="isolated")
    tok_obj.decoder = Fuse()
    if add_bos:
        # "pair" keeps <bos> first if two texts are ever encoded together.
        tok_obj.post_processor = TemplateProcessing(
            single="<bos> $A", pair="<bos> $A $B", special_tokens=[("<bos>", _VOCAB["<bos>"])]
        )
    # Wrap in a PreTrainedTokenizerFast so it behaves like any HuggingFace tokenizer (padding,
    # special tokens, save/push_to_hub, etc.).
    return PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        bos_token="<bos>",  # beginning-of-sequence
        eos_token="<eos>",  # end-of-sequence (the model learns to emit this to stop)
        pad_token="<pad>",  # fills short sequences so a batch is rectangular
        unk_token="<unk>",  # stands in for any character not in VOCAB_CHARS
        # Cleanup would remove spaces before punctuation when decoding ("a ." -> "a."), so text would
        # not decode back exactly.
        clean_up_tokenization_spaces=False,
    )


def check_tokenizer(tokenizer: PreTrainedTokenizerBase, add_bos: bool) -> None:
    """Raise AssertionError unless the tokenizer follows every rule below.

    Run on the tokenizer as training will load it (saved, then reloaded with AutoTokenizer).

    Args:
        tokenizer: The tokenizer to check.
        add_bos: Whether it should put <bos> at the start of every encoded sequence.
    """
    vocab = tokenizer.get_vocab()

    # Vocabulary structure
    expected_vocab_size = len(_SPECIAL_TOKENS) + len(VOCAB_CHARS)
    assert len(tokenizer) == expected_vocab_size, f"Expected {expected_vocab_size} tokens, got {len(tokenizer)}"

    # Special token IDs match _SPECIAL_TOKENS insertion order
    assert tokenizer.pad_token_id == 0, f"pad_token_id={tokenizer.pad_token_id}, expected 0"
    assert tokenizer.bos_token_id == 1, f"bos_token_id={tokenizer.bos_token_id}, expected 1"
    assert tokenizer.eos_token_id == 2, f"eos_token_id={tokenizer.eos_token_id}, expected 2"
    assert vocab["<unk>"] == 3, f"unk id={vocab['<unk>']}, expected 3"

    # pad and eos must be distinct, so padding can never be mistaken for the model's <eos>
    assert tokenizer.pad_token_id != tokenizer.eos_token_id, "pad_token_id must differ from eos_token_id"

    # Token IDs are unique and contiguous: every id in [0, vocab_size) is used exactly once, so the
    # model's embedding table (vocab_size rows) has no unused or shared rows.
    assert sorted(vocab.values()) == list(range(len(tokenizer))), "Token IDs are not unique and contiguous"

    # Every vocab character has its own entry and no two characters share an ID
    char_ids = [vocab[c] for c in VOCAB_CHARS]
    assert len(char_ids) == len(set(char_ids)), "Duplicate IDs among vocab characters"

    # Only the four special tokens are special: decode(skip_special_tokens=True) deletes special
    # tokens, so a vocab character marked special would silently vanish from decoded answers.
    assert sorted(tokenizer.all_special_tokens) == sorted(_SPECIAL_TOKENS), tokenizer.all_special_tokens

    # Settings the saved files must keep: padding on the right for training (evaluation switches to
    # the left itself), and no cleanup of spaces when decoding.
    assert tokenizer.padding_side == "right", f"padding_side={tokenizer.padding_side!r}, expected 'right'"
    assert not tokenizer.clean_up_tokenization_spaces, "clean_up_tokenization_spaces must be False"

    print("\nVocabulary assertions passed.")

    # Round-trip sanity checks on SAMPLE_TEXTS. For a character-level
    # tokenizer, each token should be exactly one input character.
    print("\nSanity checks:")
    for text in SAMPLE_TEXTS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        print(f"  {text!r:16s} -> {tokens}")

        assert tokens == list(text), (
            f"Tokenization mismatch for {text!r}:\n  expected: {list(text)}\n  got:      {tokens}"
        )
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert decoded == text, f"Round-trip mismatch for {text!r}:\n  expected: {text!r}\n  got:      {decoded!r}"
        assert tokenizer.decode(ids) == text, f"Round-trip mismatch for {text!r} (special tokens kept)"
        assert tokenizer.unk_token_id not in ids, f"Unknown token in encoding of {text!r}: ids={ids}"
        assert all(0 <= i < len(tokenizer) for i in ids), f"Out-of-bounds token ID in encoding of {text!r}: ids={ids}"

        # Encoding with the default special tokens (how training and inference encode prompts) adds
        # <bos> in front if, and only if, add_bos is on -- and nothing else.
        with_special = tokenizer.encode(text)
        expected = [tokenizer.bos_token_id, *ids] if add_bos else ids
        assert with_special == expected, f"Encoding {text!r} gave {with_special}, expected {expected}"
        assert tokenizer.decode(with_special, skip_special_tokens=True) == text, f"Round-trip mismatch for {text!r}"

    print("\nAll tokenization assertions passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a character-level tokenizer, check it and save it")
    parser.add_argument(
        "--hub-name",
        type=str,
        default=None,
        help="HF Hub repo to also push the tokenizer to (e.g. your-username/your-project-tokenizer), once "
        "every check has passed. If omitted, it is only saved locally to --output-dir (no login needed).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./artifacts/tokenizer",
        help="Local directory to save the tokenizer to.",
    )
    parser.add_argument(
        "--add-bos",
        action="store_true",
        help="Put a <bos> token at the start of every encoded sequence (see the module docstring).",
    )
    args = parser.parse_args()

    if not VOCAB_CHARS or not SAMPLE_TEXTS:
        raise ValueError(
            "VOCAB_CHARS and SAMPLE_TEXTS are not set. "
            "Fill these in for your task -- see the module docstring for an example."
        )

    build_tokenizer(add_bos=args.add_bos).save_pretrained(args.output_dir)

    # Check the tokenizer exactly as training and inference will load it.
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    print(f"Vocabulary ({len(tokenizer)} tokens):")
    for token, idx in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]):
        print(f"  {idx:2d}: {repr(token)}")
    check_tokenizer(tokenizer, add_bos=args.add_bos)
    print(f"\nSaved tokenizer to {args.output_dir}")

    if args.hub_name:
        tokenizer.push_to_hub(args.hub_name)
        print(f"Pushed tokenizer to hub: {args.hub_name}")
    else:
        print("(pass --hub-name to also push it to the Hub)")
