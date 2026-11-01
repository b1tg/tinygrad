import unittest
from dataclasses import replace
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig

ssm = SSMConfig(conv_kernel=4, state_size=32, group_count=1, time_step_rank=2, inner_size=16)  # the fused kernel on AMD: head_k_dim % 32
config = TransformerConfig(num_blocks=2, dim=16, hidden_dim=32, n_heads=2, n_kv_heads=1, norm_eps=1e-5, vocab_size=8, head_dim=8,
                           rope_theta=10000.0, rope_dim=8, v_head_dim=8, max_context=32, ssm_layers=(True, False), ssm=ssm, spec_drafts=2)

def random_state(model:Transformer, seed:int=0) -> dict[str, Tensor]:
  rng, state_dict = np.random.default_rng(seed), nn.state.get_state_dict(model)
  for p in state_dict.values(): p.replace(Tensor(rng.normal(size=p.shape).astype(np.float32)*0.5).cast(p.dtype)).realize()
  return state_dict

class TestSpeculativeDecoding(unittest.TestCase):
  def test_matches_greedy(self):
    spec, plain = Transformer(config), Transformer(replace(config, spec_drafts=0))
    state_dict = random_state(spec)
    nn.state.load_state_dict(plain, {k:v for k,v in state_dict.items() if not k.startswith("mtp.")}, verbose=False)
    outs = []
    for model in (spec, plain):
      first = [t for _,t in zip(range(10), model.generate(list(range(8))))]
      outs.append(first + list(model.generate(list(range(8)) + first + [2])))
    self.assertEqual(outs[0], outs[1])

  def test_hidden_state_after_accepted_drafts(self):
    spec, ref = Transformer(config), Transformer(config)
    random_state(spec, seed=3), random_state(ref, seed=3)  # a seed with accepted drafts
    jit, steps = spec.spec_jit, []
    def spec_jit(*args):
      steps.append(args)
      return jit(*args)
    spec.spec_jit = spec_jit
    gen = spec.generate(list(range(8)))
    toks = list(range(8)) + [t for _,t in zip(range(9), gen)]
    gen.close()
    self.assertLess(len(steps), 8)
    n = len(spec._cached_tokens)
    for p in range(n): ref.forward(Tensor([toks[p:p+1]]), p, Tensor([0.0])).realize()
    x = Tensor.zeros(1, 1, config.dim)
    np.testing.assert_allclose(spec.mtp.nextn.hidden_before(x, n).numpy(), ref.mtp.nextn.hidden_before(x, n).numpy(), atol=1e-2)

  def test_chunks_as_long_as_the_ring(self):
    # the slot a chunk writes last is the one it reads first
    chunked, ref = Transformer(config), Transformer(config)
    random_state(chunked, seed=3), random_state(ref, seed=3)
    toks, S = [i%config.vocab_size for i in range(12)], config.spec_drafts+2
    for p in range(0, len(toks), S): chunked.forward(Tensor([toks[p:p+S]]), p, Tensor([0.0])).realize()
    for p in range(len(toks)): ref.forward(Tensor([toks[p:p+1]]), p, Tensor([0.0])).realize()
    np.testing.assert_allclose(chunked.mtp.cache_kv[:, :, :, :len(toks)].numpy(), ref.mtp.cache_kv[:, :, :, :len(toks)].numpy(), atol=1e-2)

if __name__ == "__main__":
  unittest.main()
