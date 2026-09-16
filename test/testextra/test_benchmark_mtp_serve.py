import json, unittest
from unittest.mock import MagicMock, patch
from extra.benchmark_mtp_serve import request


class TestClientDecodeTiming(unittest.TestCase):
  def run_request(self, chunks, clock):
    response = MagicMock()
    response.status = 200
    response.__iter__.return_value = iter([b'data: '+json.dumps(c).encode()+b'\n\n' for c in chunks]+[b'data: [DONE]\n\n'])
    connection = MagicMock()
    connection.getresponse.return_value = response
    with patch('extra.benchmark_mtp_serve.http.client.HTTPConnection', return_value=connection), \
         patch('extra.benchmark_mtp_serve.time.perf_counter', side_effect=clock):
      return request(12345, 'explain rangify', 128)

  def test_buffered_tool_generation_counts_until_finish(self):
    chunks = [
      {'choices':[{'delta':{'role':'assistant', 'content':''}}]},
      {'choices':[{'delta':{'reasoning_content':'Thinking'}}]},
      {'choices':[{'delta':{'tool_calls':[{'id':'random-id', 'type':'function', 'function':{'name':'read', 'arguments':'{}'}}]}}]},
      {'choices':[{'delta':{}, 'finish_reason':'tool_calls'}]},
      {'choices':[], 'usage':{'completion_tokens':7, 'prompt_tokens':10, 'total_tokens':17}},
    ]
    result = self.run_request(chunks, [0, 1, 2, 4, 5, 6, 7])
    self.assertEqual(result['decode_seconds'], 3)
    self.assertEqual(result['tok_s'], 2)
    self.assertEqual(result['ttft'], 2)
    self.assertEqual(result['tool_calls'], [{'type':'function', 'function':{'name':'read', 'arguments':'{}'}}])

  def test_tool_only_stream_is_not_a_decode_measurement(self):
    chunks = [
      {'choices':[{'delta':{'tool_calls':[{'type':'function', 'function':{'name':'read', 'arguments':'{}'}}]}}]},
      {'choices':[{'delta':{}, 'finish_reason':'tool_calls'}]},
      {'choices':[], 'usage':{'completion_tokens':20}},
    ]
    with self.assertRaisesRegex(RuntimeError, 'cannot separate'):
      self.run_request(chunks, [0, 10, 10.001, 10.002])


if __name__ == '__main__': unittest.main()
