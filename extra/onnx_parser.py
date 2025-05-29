# https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3

import os, struct
from io import BufferedReader
from typing import Tuple
from types import SimpleNamespace
from tinygrad.nn.state import TensorIO, accept_filename, TensorIOBufferedReader
from tinygrad.tensor import Tensor, dtypes

# Protobuf Wire Types
WIRETYPE_VARINT = 0; WIRETYPE_FIXED64 = 1; WIRETYPE_LENGTH_DELIMITED = 2; WIRETYPE_START_GROUP = 3; WIRETYPE_END_GROUP = 4; WIRETYPE_FIXED32 = 5 # noqa: E702

# TensorProto.DataType
class TensorDataType:
  UNDEFINED = 0; FLOAT = 1; UINT8 = 2; INT8 = 3; UINT16 = 4; INT16 = 5; INT32 = 6; INT64 = 7 # noqa: E702
  STRING = 8; BOOL = 9; FLOAT16 = 10; DOUBLE = 11; UINT32 = 12; UINT64 = 13; COMPLEX64 = 14; COMPLEX128 = 15; BFLOAT16 = 16 # noqa: E702

# AttributeProto.AttributeType
class AttributeType:
  UNDEFINED = 0; FLOAT = 1; INT = 2; STRING = 3; TENSOR = 4; GRAPH = 5; SPARSE_TENSOR = 11; TYPE_PROTO = 13; FLOATS = 6; INTS = 7 # noqa: E702
  STRINGS = 8; TENSORS = 9; GRAPHS = 10; SPARSE_TENSORS = 12; TYPE_PROTOS = 14 # noqa: E702

@accept_filename
def onnx_load(tensor: Tensor):
  reader = TensorIOBufferedReader(TensorIO(tensor))
  parser = OnnxParser()
  onnx_model = parser.parse(reader)
  model = dict_to_namespace(onnx_model)
  return model

def gen_result(obj: dict, key_name, val, repeated: bool):
  if repeated: obj.setdefault(key_name, []).append(val)
  else: obj[key_name] = val

def dict_to_namespace(d):
  if isinstance(d, dict): return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
  elif isinstance(d, list): return [dict_to_namespace(i) for i in d]
  return d

class OnnxParser:
  def __init__(self):
    PB_INFOS = {
      "OperatorSetIdProto": {1: self.pb_string("domain"), 2: self.pb_int("version")},
      "StringStringEntryProto": {1: self.pb_string('key'), 2: self.pb_string('value')},
      "TensorProto": {1: self.pb_int("dims", True), 2: self.pb_int("data_type"), 4: self.pb_floats("float_data"), 5: self.pb_ints("int32_data"),
        7: self.pb_ints("int64_data"), 8: self.pb_string("name"), 9: self.pb_bytes("raw_data")},
      "TensorShapeProtoDimension": {1: self.pb_int('dim_value'), 2: self.pb_string('dim_param')},
      "TensorShapeProto": {1: self.pb_sub("dim", True, "TensorShapeProtoDimension")},
      "ModelProto": {1: self.pb_int("ir_version"), 2: self.pb_string("producer_name"), 3: self.pb_string("producer_version"),
        4: self.pb_string("domain"), 5: self.pb_int("model_version"), 6: self.pb_string("doc_string"),
        7: self.pb_sub("graph", False, ("GraphProto", lambda: {'node': [], 'initializer': [], 'input': [], 'output': [], 'value_info': []})),
        8: self.pb_sub("opset_import", True, "OperatorSetIdProto")},
      "GraphProto": {1: self.pb_sub("node", True, ("NodeProto", lambda: {'input': [], 'output': [], 'attribute': [], 'domain': None})),
        2: self.pb_string("name"), 10: self.pb_string("doc_string"),
        5: self.pb_sub("initializer", True, ("TensorProto", lambda: {'dims': [], 'float_data': [], 'int32_data': [], 'string_data': [],
                                                                     'int64_data': [], 'double_data': [], 'uint64_data': []})),
        11: self.pb_sub("input", True, "ValueInfoProto"), 12: self.pb_sub("output", True, "ValueInfoProto")},
      "NodeProto": {1: self.pb_string("input", True), 2: self.pb_string("output", True), 3: self.pb_string("name"), 4: self.pb_string("op_type"),
        5: self.pb_sub("attribute", True, ("AttributeProto", lambda: {'floats': [], 'ints': [], 'strings': []})),
        6: self.pb_string("doc_string"), 7: self.pb_string("domain")},
      "AttributeProto": {1: self.pb_string("name"), 20: self.pb_int("type"), 3: self.pb_int("i"), 8: self.pb_int("ints", True),
        2: self.pb_float("f"), 7: self.pb_float("floats", True), 4: self.pb_string("s"), 9: self.pb_string("strings", True),
        5: self.pb_sub("t", False, ("TensorProto", lambda: {'dims': [], 'float_data': [], 'int32_data': [], 'string_data': [],
                                                            'int64_data': [], 'double_data': [], 'uint64_data': []}))},
      "ValueInfoProto": {1: self.pb_string('name'), 2: self.pb_sub('type', False, "TypeProto"), 3: self.pb_string('doc_string')},
      "TypeProto": {1: self.pb_sub('tensor_type', False, "TypeProtoTensor"), 4: self.pb_sub('sequence_type', False, "TypeProtoSequence"),
        9: self.pb_sub('optional_type', False, "TypeProtoOptional"), 6: self.pb_string('denotation')},
      "TypeProtoSequence": {1: self.pb_sub('elem_type', False, "TypeProto")}, "TypeProtoOptional": {1: self.pb_sub('elem_type', False, "TypeProto")},
      "TypeProtoTensor": {1: self.pb_int('elem_type'), 2: self.pb_sub('shape', False, ("TensorShapeProto", lambda: {'dim': []}))}
    }
    self.registered_handles = {}
    for pb_name in PB_INFOS:
      handler_map = {}
      for fid, config in PB_INFOS[pb_name].items():
        name, handler_fn, repeated, parser_fn = config
        def _wrapper_handler(obj, reader, wt, h=handler_fn, n=name, p=parser_fn, r=repeated): return h(obj, n, reader, wt, parser_func=p, repeated=r)
        _wrapper_handler._debug_info = f"{fid}, {name} => {handler_fn}"
        handler_map[fid] = _wrapper_handler
      self.registered_handles[pb_name] = handler_map

  def parse(self, reader): return self._parse_message(reader, "ModelProto", lambda: {'opset_import': [], 'domain': None, 'graph': None})

  def pb_bytes(self, name, repeat=False): return name, self._handle_bytes, repeat, None
  def pb_float(self, name, repeat=False): return name, self._handle_float, repeat, None
  def pb_floats(self, name, repeat=False): return name, self._handle_packed_floats, repeat, None
  def pb_int(self, name, repeat=False): return name, self._handle_int64, repeat, None
  def pb_ints(self, name, repeat=False): return name, self._handle_packed_int64s, repeat, None
  def pb_string(self, name, repeat=False): return name, self._handle_string, repeat, None
  def pb_sub(self, name, repeat=False, parser_fn=None): return name, self._handle_sub_message, repeat, parser_fn

  def decode_varint(self, reader: BufferedReader) -> int:
    result = 0
    shift = 0
    while True:
      data = reader.read(1)
      if data == b'': raise EOFError("decode_varint EOF")
      result |= (data[0] & 0x7F) << shift
      if not (data[0] & 0x80): return result
      shift += 7
      if shift >= 70: raise ValueError("Varint too long")

  def skip_field_value(self, reader: BufferedReader, wire_type):
    if wire_type == WIRETYPE_VARINT: self.decode_varint(reader)
    elif wire_type == WIRETYPE_FIXED64: reader.seek(8, os.SEEK_CUR)
    elif wire_type == WIRETYPE_FIXED32: reader.seek(4, os.SEEK_CUR)
    elif wire_type == WIRETYPE_LENGTH_DELIMITED: reader.seek(self.decode_varint(reader), os.SEEK_CUR)
    else: raise ValueError(f"Unknown wire type: {wire_type}")

  def _parse_message(self, reader, name, initial_obj_factory=lambda: {}):
    message_field_handlers = self.registered_handles[name]
    obj = initial_obj_factory()
    while True:
      try:
        tag_val = self.decode_varint(reader)
        field_number = tag_val >> 3
        wire_type = tag_val & 0x07
        if handler := message_field_handlers.get(field_number):
          handler(obj, reader, wire_type)
        else: self.skip_field_value(reader, wire_type)
      except EOFError: break
    return obj

  def _handle_delimited(self, reader:TensorIOBufferedReader, use_tensor=False) -> Tuple[bytes, Tensor]:
    str_len = self.decode_varint(reader)
    if not use_tensor: return reader.read(str_len)
    res = reader.raw._tensor[reader.tell():(reader.tell()+str_len)] # TODO: hack
    reader.seek(str_len, os.SEEK_CUR)
    return res

  def _handle_string(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_LENGTH_DELIMITED: raise ValueError(f"Expected length-delimited for string field '{key_name}'")
    value = self._handle_delimited(reader)
    gen_result(obj, key_name, value.decode('utf-8'), repeated)

  def _handle_bytes(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_LENGTH_DELIMITED: raise ValueError(f"Expected length-delimited for bytes field '{key_name}'")
    value = self._handle_delimited(reader, use_tensor=True)
    gen_result(obj, key_name, value, repeated)

  def _handle_int64(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_VARINT: raise ValueError(f"Expected varint for int64 field '{key_name}'")
    val = self.decode_varint(reader)
    gen_result(obj, key_name, val - 2**64 if val & (1 << 63) else val, repeated)

  def _handle_float(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_FIXED32: raise ValueError(f"Expected fixed32 for float field '{key_name}'")
    val, = struct.unpack("<f", reader.read(4))
    gen_result(obj, key_name, val, repeated)

  def _handle_packed_int64s(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_LENGTH_DELIMITED: raise ValueError("Packed int64s expected length_delimited")
    total_bytes_len = self.decode_varint(reader)
    old_pos = reader.tell()
    values = []
    while reader.tell() < total_bytes_len + old_pos:
      val = self.decode_varint(reader)
      values.append(val - 2**64 if val & (1 << 63) else val)
    obj[key_name] = values

  def _handle_packed_floats(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_LENGTH_DELIMITED: raise ValueError("Packed floats expected length_delimited")
    value = self._handle_delimited(reader, use_tensor=True)
    obj[key_name] = value.bitcast(dtypes.float32)

  def _handle_sub_message(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_LENGTH_DELIMITED: raise ValueError(f"Expected length-delimited for sub-message field '{key_name}'")
    value = self._handle_delimited(reader, use_tensor=True)
    if isinstance(parser_func, str): sub_obj = self._parse_message(TensorIOBufferedReader(TensorIO(value)), parser_func)
    elif isinstance(parser_func, tuple): sub_obj = self._parse_message(TensorIOBufferedReader(TensorIO(value)), parser_func[0], parser_func[1])
    else: raise Exception("no parser_func for sub_message handle")
    gen_result(obj, key_name, sub_obj, repeated)
