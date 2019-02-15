/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/util.h"

namespace tflite {
namespace ops {

namespace custom {

TfLiteRegistration* Register_AUDIO_SPECTROGRAM();
TfLiteRegistration* Register_LAYER_NORM_LSTM();
TfLiteRegistration* Register_MFCC();
TfLiteRegistration* Register_DETECTION_POSTPROCESS();
TfLiteRegistration* Register_RELU_1();

}  // namespace custom

namespace builtin {

TfLiteRegistration* Register_RELU();
TfLiteRegistration* Register_RELU_N1_TO_1();
TfLiteRegistration* Register_RELU6();
TfLiteRegistration* Register_TANH();
TfLiteRegistration* Register_LOGISTIC();
TfLiteRegistration* Register_AVERAGE_POOL_2D();
TfLiteRegistration* Register_MAX_POOL_2D();
TfLiteRegistration* Register_L2_POOL_2D();
TfLiteRegistration* Register_CONV_2D();
TfLiteRegistration* Register_DEPTHWISE_CONV_2D();
TfLiteRegistration* Register_SVDF();
TfLiteRegistration* Register_RNN();
TfLiteRegistration* Register_BIDIRECTIONAL_SEQUENCE_RNN();
TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_RNN();
TfLiteRegistration* Register_EMBEDDING_LOOKUP();
TfLiteRegistration* Register_EMBEDDING_LOOKUP_SPARSE();
TfLiteRegistration* Register_FULLY_CONNECTED();
TfLiteRegistration* Register_LSH_PROJECTION();
TfLiteRegistration* Register_HASHTABLE_LOOKUP();
TfLiteRegistration* Register_SOFTMAX();
TfLiteRegistration* Register_CONCATENATION();
TfLiteRegistration* Register_ADD();
TfLiteRegistration* Register_SPACE_TO_BATCH_ND();
TfLiteRegistration* Register_DIV();
TfLiteRegistration* Register_SUB();
TfLiteRegistration* Register_BATCH_TO_SPACE_ND();
TfLiteRegistration* Register_MUL();
TfLiteRegistration* Register_L2_NORMALIZATION();
TfLiteRegistration* Register_LOCAL_RESPONSE_NORMALIZATION();
TfLiteRegistration* Register_LSTM();
TfLiteRegistration* Register_BIDIRECTIONAL_SEQUENCE_LSTM();
TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
TfLiteRegistration* Register_PAD();
TfLiteRegistration* Register_PADV2();
TfLiteRegistration* Register_RESHAPE();
TfLiteRegistration* Register_RESIZE_BILINEAR();
TfLiteRegistration* Register_SKIP_GRAM();
TfLiteRegistration* Register_SPACE_TO_DEPTH();
TfLiteRegistration* Register_GATHER();
TfLiteRegistration* Register_TRANSPOSE();
TfLiteRegistration* Register_MEAN();
TfLiteRegistration* Register_SPLIT();
TfLiteRegistration* Register_SQUEEZE();
TfLiteRegistration* Register_STRIDED_SLICE();
TfLiteRegistration* Register_EXP();
TfLiteRegistration* Register_TOPK_V2();
TfLiteRegistration* Register_LOG();
TfLiteRegistration* Register_LOG_SOFTMAX();
TfLiteRegistration* Register_CAST();
TfLiteRegistration* Register_DEQUANTIZE();
TfLiteRegistration* Register_PRELU();
TfLiteRegistration* Register_MAXIMUM();
TfLiteRegistration* Register_MINIMUM();
TfLiteRegistration* Register_ARG_MAX();
TfLiteRegistration* Register_ARG_MIN();
TfLiteRegistration* Register_GREATER();
TfLiteRegistration* Register_GREATER_EQUAL();
TfLiteRegistration* Register_LESS();
TfLiteRegistration* Register_LESS_EQUAL();
TfLiteRegistration* Register_FLOOR();
TfLiteRegistration* Register_TILE();
TfLiteRegistration* Register_NEG();
TfLiteRegistration* Register_SUM();
TfLiteRegistration* Register_REDUCE_PROD();
TfLiteRegistration* Register_REDUCE_MAX();
TfLiteRegistration* Register_REDUCE_MIN();
TfLiteRegistration* Register_REDUCE_ANY();
TfLiteRegistration* Register_SELECT();
TfLiteRegistration* Register_SLICE();
TfLiteRegistration* Register_SIN();
TfLiteRegistration* Register_TRANSPOSE_CONV();
TfLiteRegistration* Register_EXPAND_DIMS();
TfLiteRegistration* Register_SPARSE_TO_DENSE();
TfLiteRegistration* Register_EQUAL();
TfLiteRegistration* Register_NOT_EQUAL();
TfLiteRegistration* Register_SQRT();
TfLiteRegistration* Register_RSQRT();
TfLiteRegistration* Register_SHAPE();
TfLiteRegistration* Register_POW();
TfLiteRegistration* Register_FAKE_QUANT();
TfLiteRegistration* Register_PACK();
TfLiteRegistration* Register_ONE_HOT();
TfLiteRegistration* Register_LOGICAL_OR();
TfLiteRegistration* Register_LOGICAL_AND();
TfLiteRegistration* Register_LOGICAL_NOT();
TfLiteRegistration* Register_UNPACK();
TfLiteRegistration* Register_FLOOR_DIV();
TfLiteRegistration* Register_SQUARE();
TfLiteRegistration* Register_ZEROS_LIKE();

TfLiteStatus UnsupportedTensorFlowOp(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(
      context,
      "Regular TensorFlow ops are not supported by this interpreter. Make sure "
      "you invoke the Flex delegate before inference.");
  return kTfLiteError;
}

const TfLiteRegistration* BuiltinOpResolver::FindOp(tflite::BuiltinOperator op,
                                                    int version) const {
  return MutableOpResolver::FindOp(op, version);
}

const TfLiteRegistration* BuiltinOpResolver::FindOp(const char* op,
                                                    int version) const {
  // Return the NULL Op for all ops whose name start with "Flex", allowing
  // the interpreter to delegate their execution.
  if (IsFlexOp(op)) {
    static TfLiteRegistration null_op{
        nullptr, nullptr, &UnsupportedTensorFlowOp,
        nullptr, nullptr, BuiltinOperator_CUSTOM,
        "Flex",  1};
    return &null_op;
  }
  return MutableOpResolver::FindOp(op, version);
}

BuiltinOpResolver::BuiltinOpResolver() {
  AddBuiltin(BuiltinOperator_RELU, Register_RELU(), "RELU");
  AddBuiltin(BuiltinOperator_RELU_N1_TO_1, Register_RELU_N1_TO_1(), "RELU_N1_TO_1");
  AddBuiltin(BuiltinOperator_RELU6, Register_RELU6(), "RELU6");
  AddBuiltin(BuiltinOperator_TANH, Register_TANH(), "TANH");
  AddBuiltin(BuiltinOperator_LOGISTIC, Register_LOGISTIC(), "LOGISTIC");
  AddBuiltin(BuiltinOperator_AVERAGE_POOL_2D, Register_AVERAGE_POOL_2D(), "AVERAGE_POOL_2D");
  AddBuiltin(BuiltinOperator_MAX_POOL_2D, Register_MAX_POOL_2D(), "MAX_POOL_2D");
  AddBuiltin(BuiltinOperator_L2_POOL_2D, Register_L2_POOL_2D(), "L2_POOL_2D");
  AddBuiltin(BuiltinOperator_CONV_2D, Register_CONV_2D(), "CONV_2D");
  AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, Register_DEPTHWISE_CONV_2D(), "DEPTHWISE_CONV_2D",
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_SVDF, Register_SVDF(), "SVDF");
  AddBuiltin(BuiltinOperator_RNN, Register_RNN(), "RNN");
  AddBuiltin(BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN,
             Register_BIDIRECTIONAL_SEQUENCE_RNN(), "BIDIRECTIONAL_SEQUENCE_RNN");
  AddBuiltin(BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN,
             Register_UNIDIRECTIONAL_SEQUENCE_RNN(), "UNIDIRECTIONAL_SEQUENCE_RNN");
  AddBuiltin(BuiltinOperator_EMBEDDING_LOOKUP, Register_EMBEDDING_LOOKUP(), "EMBEDDING_LOOKUP");
  AddBuiltin(BuiltinOperator_EMBEDDING_LOOKUP_SPARSE,
             Register_EMBEDDING_LOOKUP_SPARSE(), "EMBEDDING_LOOKUP_SPARSE");
             // Register_EMBEDDING_LOOKUP_SPARSE());
  AddBuiltin(BuiltinOperator_FULLY_CONNECTED, Register_FULLY_CONNECTED(), "FULLY_CONNECTED",
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_LSH_PROJECTION, Register_LSH_PROJECTION(), "LSH_PROJECTION");
  AddBuiltin(BuiltinOperator_HASHTABLE_LOOKUP, Register_HASHTABLE_LOOKUP(), "HASHTABLE_LOOKUP");
  AddBuiltin(BuiltinOperator_SOFTMAX, Register_SOFTMAX(), "SOFTMAX");
  AddBuiltin(BuiltinOperator_CONCATENATION, Register_CONCATENATION(), "CONCATENATION");
  AddBuiltin(BuiltinOperator_ADD, Register_ADD(), "ADD");
  AddBuiltin(BuiltinOperator_SPACE_TO_BATCH_ND, Register_SPACE_TO_BATCH_ND(), "SPACE_TO_BATCH_ND");
  AddBuiltin(BuiltinOperator_BATCH_TO_SPACE_ND, Register_BATCH_TO_SPACE_ND(), "BATCH_TO_SPACE_ND");
  AddBuiltin(BuiltinOperator_MUL, Register_MUL(), "MUL");
  AddBuiltin(BuiltinOperator_L2_NORMALIZATION, Register_L2_NORMALIZATION(), "L2_NORMALIZATION");
  AddBuiltin(BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION,
             Register_LOCAL_RESPONSE_NORMALIZATION(), "LOCAL_RESPONSE_NORMALIZATION");
  AddBuiltin(BuiltinOperator_LSTM, Register_LSTM(), "LSTM",
             /* min_version */ 1,
             /* max_version */ 2);
#if 0
// >>>>>>> origin/ledl-baseline
  AddBuiltin(BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM,
             Register_BIDIRECTIONAL_SEQUENCE_LSTM(), "BIDIRECTIONAL_SEQUENCE_LSTM");
  AddBuiltin(BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM,
             Register_UNIDIRECTIONAL_SEQUENCE_LSTM(), "UNIDIRECTIONAL_SEQUENCE_LSTM");
             // Register_UNIDIRECTIONAL_SEQUENCE_LSTM());
#endif
  AddBuiltin(BuiltinOperator_PAD, Register_PAD(), "PAD");
  AddBuiltin(BuiltinOperator_PADV2, Register_PADV2(), "PADV2");
  AddBuiltin(BuiltinOperator_RESHAPE, Register_RESHAPE(), "RESHAPE");
  AddBuiltin(BuiltinOperator_RESIZE_BILINEAR, Register_RESIZE_BILINEAR(), "RESIZE_BILINEAR");
  AddBuiltin(BuiltinOperator_SKIP_GRAM, Register_SKIP_GRAM(), "SKIP_GRAM");
  AddBuiltin(BuiltinOperator_SPACE_TO_DEPTH, Register_SPACE_TO_DEPTH(), "SPACE_TO_DEPTH");
  AddBuiltin(BuiltinOperator_GATHER, Register_GATHER(), "GATHER");
  AddBuiltin(BuiltinOperator_TRANSPOSE, Register_TRANSPOSE(), "TRANSPOSE");
  AddBuiltin(BuiltinOperator_MEAN, Register_MEAN(), "MEAN");
  AddBuiltin(BuiltinOperator_DIV, Register_DIV(), "DIV");
  AddBuiltin(BuiltinOperator_SUB, Register_SUB(), "SUB");
  AddBuiltin(BuiltinOperator_SPLIT, Register_SPLIT(), "SPLIT");
  AddBuiltin(BuiltinOperator_SQUEEZE, Register_SQUEEZE(), "SQUEEZE");
  AddBuiltin(BuiltinOperator_STRIDED_SLICE, Register_STRIDED_SLICE(), "STRIDED_SLICE");
  AddBuiltin(BuiltinOperator_EXP, Register_EXP(), "EXP");
  AddBuiltin(BuiltinOperator_TOPK_V2, Register_TOPK_V2(), "TOPK_V2");
  AddBuiltin(BuiltinOperator_LOG_SOFTMAX, Register_LOG_SOFTMAX(), "LOG_SOFTMAX");
  AddBuiltin(BuiltinOperator_CAST, Register_CAST(), "CAST");
  AddBuiltin(BuiltinOperator_DEQUANTIZE, Register_DEQUANTIZE(), "DEQUANTIZE");
  AddBuiltin(BuiltinOperator_PRELU, Register_PRELU(), "PRELU");
  AddBuiltin(BuiltinOperator_MAXIMUM, Register_MAXIMUM(), "MAXIMUM");

  AddBuiltin(BuiltinOperator_MINIMUM, Register_MINIMUM(), "MINIMUM");
  AddBuiltin(BuiltinOperator_ARG_MAX, Register_ARG_MAX(), "ARG_MAX");
  AddBuiltin(BuiltinOperator_ARG_MIN, Register_ARG_MIN(), "ARG_MIN");
  AddBuiltin(BuiltinOperator_GREATER, Register_GREATER(), "GREATER");
  AddBuiltin(BuiltinOperator_GREATER_EQUAL, Register_GREATER_EQUAL(), "GREATER_EQUAL");
  AddBuiltin(BuiltinOperator_LESS, Register_LESS(), "LESS");
  AddBuiltin(BuiltinOperator_LESS_EQUAL, Register_LESS_EQUAL(), "LESS_EQUAL");
  AddBuiltin(BuiltinOperator_FLOOR, Register_FLOOR(), "FLOOR");
  AddBuiltin(BuiltinOperator_NEG, Register_NEG(), "NEG");
  AddBuiltin(BuiltinOperator_SELECT, Register_SELECT(), "SELECT");
  AddBuiltin(BuiltinOperator_SLICE, Register_SLICE(), "SLICE");
  AddBuiltin(BuiltinOperator_SIN, Register_SIN(), "SIN");
  AddBuiltin(BuiltinOperator_TRANSPOSE_CONV, Register_TRANSPOSE_CONV(), "TRANSPOSE_CONV");
  AddBuiltin(BuiltinOperator_TILE, Register_TILE(), "TILE");
  AddBuiltin(BuiltinOperator_SUM, Register_SUM(), "SUM");
  AddBuiltin(BuiltinOperator_REDUCE_PROD, Register_REDUCE_PROD(), "REDUCE_PROD");
  AddBuiltin(BuiltinOperator_REDUCE_MAX, Register_REDUCE_MAX(), "REDUCE_MAX");
  AddBuiltin(BuiltinOperator_REDUCE_MIN, Register_REDUCE_MIN(), "REDUCE_MIN");
  AddBuiltin(BuiltinOperator_REDUCE_ANY, Register_REDUCE_ANY(), "REDUCE_ANY");
  AddBuiltin(BuiltinOperator_EXPAND_DIMS, Register_EXPAND_DIMS(), "EXPAND_DIMS");
  AddBuiltin(BuiltinOperator_SPARSE_TO_DENSE, Register_SPARSE_TO_DENSE(), "SPARSE_TO_DENSE");
  AddBuiltin(BuiltinOperator_EQUAL, Register_EQUAL(), "EQUAL");
  AddBuiltin(BuiltinOperator_NOT_EQUAL, Register_NOT_EQUAL(), "NOT_EQUAL");
  AddBuiltin(BuiltinOperator_SQRT, Register_SQRT(), "SQRT");
  AddBuiltin(BuiltinOperator_RSQRT, Register_RSQRT(), "RSQRT");
  AddBuiltin(BuiltinOperator_SHAPE, Register_SHAPE(), "SHAPE");
  AddBuiltin(BuiltinOperator_POW, Register_POW(), "POW");
  // Fake quant
  AddBuiltin(BuiltinOperator_FAKE_QUANT, Register_FAKE_QUANT(), "FAKE_QUANT",
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_PACK, Register_PACK(), "PACK");
  AddBuiltin(BuiltinOperator_ONE_HOT, Register_ONE_HOT(), "ONE_HOT");
  AddBuiltin(BuiltinOperator_LOGICAL_OR, Register_LOGICAL_OR(), "LOGICAL_OR");
  AddBuiltin(BuiltinOperator_LOGICAL_AND, Register_LOGICAL_AND(), "LOGICAL_AND");
  AddBuiltin(BuiltinOperator_LOGICAL_NOT, Register_LOGICAL_NOT(), "LOGICAL_NOT");
  AddBuiltin(BuiltinOperator_UNPACK, Register_UNPACK(), "UNPACK");
  AddBuiltin(BuiltinOperator_FLOOR_DIV, Register_FLOOR_DIV(), "FLOOR_DIV");
  AddBuiltin(BuiltinOperator_SQUARE, Register_SQUARE(), "SQUARE");
  AddBuiltin(BuiltinOperator_ZEROS_LIKE, Register_ZEROS_LIKE(), "ZEROS_LIKE");
#if 0
  // TODO(andrewharp, ahentz): Move these somewhere more appropriate so that
  // custom ops aren't always included by default.
  AddCustom("Mfcc", tflite::ops::custom::Register_MFCC());
  AddCustom("AudioSpectrogram",
            tflite::ops::custom::Register_AUDIO_SPECTROGRAM());
// <<<<<<< HEAD
  AddCustom("LayerNormLstm", tflite::ops::custom::Register_LAYER_NORM_LSTM());
  AddCustom("Relu1", tflite::ops::custom::Register_RELU_1());
  AddCustom("TFLite_Detection_PostProcess",
            tflite::ops::custom::Register_DETECTION_POSTPROCESS());
// =======
#endif
}

TfLiteRegistration* BuiltinOpResolver::FindOp(
    tflite::BuiltinOperator op) const {
  auto it = builtins_.find(op);
  return it != builtins_.end() ? it->second : nullptr;
}

TfLiteRegistration* BuiltinOpResolver::FindOp(const char* op) const {
  auto it = custom_ops_.find(op);
  return it != custom_ops_.end() ? it->second : nullptr;
}

// >>>>>>> origin/ledl-baseline

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
