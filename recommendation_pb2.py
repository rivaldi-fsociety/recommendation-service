# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: recommendation.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'recommendation.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x14recommendation.proto\x12\x0erecommendation\"\x1e\n\x0bUserRequest\x12\x0f\n\x07user_id\x18\x01 \x01(\t\"3\n\x16RecommendationResponse\x12\x19\n\x11recommended_items\x18\x01 \x03(\t2r\n\x15RecommendationService\x12Y\n\x12GetRecommendations\x12\x1b.recommendation.UserRequest\x1a&.recommendation.RecommendationResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'recommendation_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_USERREQUEST']._serialized_start=40
  _globals['_USERREQUEST']._serialized_end=70
  _globals['_RECOMMENDATIONRESPONSE']._serialized_start=72
  _globals['_RECOMMENDATIONRESPONSE']._serialized_end=123
  _globals['_RECOMMENDATIONSERVICE']._serialized_start=125
  _globals['_RECOMMENDATIONSERVICE']._serialized_end=239
# @@protoc_insertion_point(module_scope)
