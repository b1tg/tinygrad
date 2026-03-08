import unittest
from unittest.mock import patch, MagicMock
from tinygrad import Tensor
from tinygrad.apps.llm import Transformer

class TestGenerateTokenTruncation(unittest.TestCase):
  def test_tokens_longer_than_max_context(self):
    """generate() should not crash when len(tokens) > max_context. It should truncate to the last max_context tokens."""
    max_context = 8

    model = Transformer(num_blocks=1, dim=32, hidden_dim=64, n_heads=2, n_kv_heads=2,
                        norm_eps=1e-5, vocab_size=100, head_dim=16, rope_theta=10000.0, max_context=max_context)

    # tokens longer than max_context -- this used to crash with "size mismatch" on reshape
    tokens = list(range(max_context + 5))

    # We just need to verify generate() doesn't crash on setup. We don't need to run the full
    # generation loop, so we grab only the first token and stop.
    gen = model.generate(tokens, chunk_size=4)
    try:
      first_tok = next(gen)
      # If we get here, the fix works -- generate didn't crash
      self.assertIsInstance(first_tok, int)
    except StopIteration:
      # Also acceptable -- model finished immediately (max_context reached)
      pass

  def test_tokens_equal_to_max_context(self):
    """generate() should handle len(tokens) == max_context (boundary case) without crashing."""
    max_context = 8
    model = Transformer(num_blocks=1, dim=32, hidden_dim=64, n_heads=2, n_kv_heads=2,
                        norm_eps=1e-5, vocab_size=100, head_dim=16, rope_theta=10000.0, max_context=max_context)

    tokens = list(range(max_context))
    gen = model.generate(tokens, chunk_size=4)
    # Should not crash. With exactly max_context tokens, the while loop condition
    # `len(tokens) < self.max_context` is already false, so no tokens are generated.
    result = list(gen)
    self.assertEqual(result, [])

  def test_tokens_shorter_than_max_context(self):
    """generate() should work normally when len(tokens) < max_context."""
    max_context = 16
    model = Transformer(num_blocks=1, dim=32, hidden_dim=64, n_heads=2, n_kv_heads=2,
                        norm_eps=1e-5, vocab_size=100, head_dim=16, rope_theta=10000.0, max_context=max_context)

    tokens = [1, 2, 3]
    gen = model.generate(tokens, chunk_size=4)
    first_tok = next(gen)
    self.assertIsInstance(first_tok, int)

if __name__ == "__main__":
  unittest.main()
