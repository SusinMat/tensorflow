#include "tensorflow/contrib/lite/kernels/user_ops/user_ops.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/padding.h"

#include "flatbuffers/flexbuffers.h" // TF:flatbuffers

namespace tflite {
namespace ops {
namespace dragunov {

#define   DIMS_1(shape) shape.Dims(1)
#define   DIMS_2(shape) shape.Dims(1) * shape.Dims(2)
#define   DIMS_3(shape) shape.Dims(1) * shape.Dims(2) * shape.Dims(3)
#define   DIMS_4(shape) shape.Dims(1) * shape.Dims(2) * shape.Dims(3) * shape.Dims(4)
#define   DIMS_5(shape) shape.Dims(1) * shape.Dims(2) * shape.Dims(3) * shape.Dims(4) * shape.Dims(5)
#define   DIMS_6(shape) shape.Dims(1) * shape.Dims(2) * shape.Dims(3) * shape.Dims(4) * shape.Dims(5) * shape.Dims(6)
#define INDEX_2D(shape, i, j)             (i * DIMS_1(shape) + j)
#define INDEX_3D(shape, i, j, k)          (i * DIMS_2(shape) + j * DIMS_1(shape) + k)
#define INDEX_4D(shape, i, j, k, l)       (i * DIMS_3(shape) + j * DIMS_2(shape) + k * DIMS_1(shape) + l)
// INDEX_4D can be replaced with SubscriptToIndex from tensorflow/contrib/lite/kernels/internal/common.h
//                       or with Offset           from tensorflow/contrib/lite/kernels/internal/types.h
#define INDEX_5D(shape, i, j, k, l, m)    (i * DIMS_4(shape) + j * DIMS_3(shape) + k * DIMS_2(shape) + l * DIMS_1(shape) + m)
#define INDEX_6D(shape, i, j, k, l, m, n) (i * DIMS_5(shape) + j * DIMS_4(shape) + k * DIMS_3(shape) + l * DIMS_2(shape) + m * DIMS_1(shape) + n)

struct TfLiteDragunovParams {
  int stride_h;
  int stride_w;
  TfLitePadding padding_type;
  TfLitePaddingValues padding_values;
  TfLiteFusedActivation activation;
};

const int dilation_height_factor = 1;
const int dilation_width_factor = 1;

constexpr int INPUT_TENSOR = 0;
constexpr int C_TENSOR = 1;
constexpr int Z_TENSOR = 2;
constexpr int F_TENSOR = 3;
constexpr int ICLUST_TENSOR = 4;
constexpr int OCLUST_TENSOR = 5;
constexpr int BIAS_TENSOR = 6;
constexpr int OUTPUT_TENSOR = 0;

inline PaddingType RuntimePaddingType(TfLitePadding padding_type) {
  switch (padding_type) {
    case TfLitePadding::kTfLitePaddingSame:
      return PaddingType::kSame;
    case TfLitePadding::kTfLitePaddingValid:
      return PaddingType::kValid;
    case TfLitePadding::kTfLitePaddingUnknown:
    default:
      return PaddingType::kNone;
  }
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  struct TfLiteDragunovParams *data = new struct TfLiteDragunovParams;

  const uint8_t *buffer_t = reinterpret_cast<const uint8_t *>(buffer);

  const flexbuffers::Map &m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  data->stride_h = m["stride_h"].AsInt64();
  data->stride_w = m["stride_w"].AsInt64();
  data->padding_type = (TfLitePadding)m["padding"].AsInt64();
  data->activation = TfLiteFusedActivation(m["fused_activation_function"].AsInt64());
  // TODO: this needs to be a parameter

  return data;
}

void Free(TfLiteContext *context, void *buffer) {
  delete reinterpret_cast<struct TfLiteDragunovParams *>(buffer);
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node)
{
  // TODO: read kernels/conv.cc and implement the transposition to HWCN weights
  // TODO: 53 out of 94 conv in inception_v3 need im2col
  struct TfLiteDragunovParams *params = reinterpret_cast<TfLiteDragunovParams *>(node->user_data);

  printf("Strides [h, w]: [%d, %d]\n", params->stride_h, params->stride_w);
  printf("Padding type: ");

  switch(params->padding_type) {
    case 0:
      printf("UNKNOWN(0)\n");
      break;
    case 1:
      printf("SAME(1)\n");
      break;
    case 2:
      printf("VALID(2)\n");
      break;
    default:
      printf("ERROR(%d)\n", params->padding_type);
      break;
  }

	TF_LITE_ENSURE_EQ(context, NumInputs(node), 7);
	TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

	const TfLiteTensor *input = GetInput(context, node, INPUT_TENSOR);
	const TfLiteTensor *f_filter = GetInput(context, node, F_TENSOR);
	const TfLiteTensor *z_filter = GetInput(context, node, Z_TENSOR);
	TfLiteTensor *output = GetOutput(context, node, OUTPUT_TENSOR);

  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  int in_height = input->dims->data[1], in_width = input->dims->data[2];
  int filter_height = z_filter->dims->data[1], filter_width = z_filter->dims->data[2];
  int out_height = -1, out_width = -1;

  if (params->padding_type == kTfLitePaddingSame) {
    out_height = (int)ceilf(float(in_height) / float(params->stride_h));
    out_width  = (int)ceilf(float(in_width) / float(params->stride_w));
  } else if (params->padding_type == kTfLitePaddingValid) {
    out_height = (int)ceilf(float(in_height - filter_height + 1) / float(params->stride_h));
    out_width  = (int)ceilf(float(in_width - filter_width + 1) / float(params->stride_w));
  }

  params->padding_values.height = ComputePadding(params->stride_h, dilation_height_factor, in_height, filter_height, out_height);
  params->padding_values.width = ComputePadding(params->stride_w, dilation_width_factor, in_width, filter_width, out_width);

	TfLiteIntArray *output_size = TfLiteIntArrayCreate(4);

  output_size->data[0] = 1;
  output_size->data[1] = out_height;
  output_size->data[2] = out_width;
  output_size->data[3] = f_filter->dims->data[0] * f_filter->dims->data[3];

  TfLiteStatus retval = context->ResizeTensor(context, output, output_size);

  return retval;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node)
{
  struct TfLiteDragunovParams *params = reinterpret_cast<TfLiteDragunovParams *>(node->user_data);
	const TfLiteTensor *input = GetInput(context, node, INPUT_TENSOR);
	const TfLiteTensor *iclust = GetInput(context, node, ICLUST_TENSOR);
	const TfLiteTensor *oclust = GetInput(context, node, OCLUST_TENSOR);
	const TfLiteTensor *c_filter = GetInput(context, node, C_TENSOR);
	const TfLiteTensor *z_filter = GetInput(context, node, Z_TENSOR);
	const TfLiteTensor *f_filter = GetInput(context, node, F_TENSOR);
	const TfLiteTensor *bias = GetInput(context, node, BIAS_TENSOR);
	TfLiteTensor *output = GetOutput(context, node, OUTPUT_TENSOR);

	const float *input_data = GetTensorData<float>(input);
	const int *iclust_data = GetTensorData<int>(iclust);
	const int *oclust_data = GetTensorData<int>(oclust);
	float *output_data = GetTensorData<float>(output);
  const RuntimeShape output_shape = GetTensorShape(output);
  const RuntimeShape iclust_shape = GetTensorShape(iclust);
  const RuntimeShape oclust_shape = GetTensorShape(oclust);
  const RuntimeShape c_filter_shape = GetTensorShape(c_filter);
  const RuntimeShape z_filter_shape = GetTensorShape(z_filter);
  const RuntimeShape f_filter_shape = GetTensorShape(f_filter);

  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min, &output_activation_max);

  printf("C filter: ");
  for (int i = 0; i < c_filter_shape.DimensionsCount(); ++i)
    printf("%d ", c_filter_shape.Dims(i));
  printf("\n");

  printf("Z filter: ");
  for (int i = 0; i < z_filter_shape.DimensionsCount(); ++i)
    printf("%d ", z_filter_shape.Dims(i));
  printf("\n");

  printf("F filter: ");
  for (int i = 0; i < f_filter_shape.DimensionsCount(); ++i)
    printf("%d ", f_filter_shape.Dims(i));
  printf("\n");

  ConvParams op_params;

  // Phase C
  // im2col not required yet, unless (stride_h, stride_w) != (1, 1)
  // hwcn is always required
  op_params.padding_type = RuntimePaddingType(params->padding_type);
  op_params.padding_values.height = params->padding_values.height;
  op_params.padding_values.width = params->padding_values.width;
  op_params.stride_height = params->stride_h;
  op_params.stride_width = params->stride_w;
  op_params.dilation_height_factor = dilation_height_factor;
  op_params.dilation_width_factor = dilation_width_factor;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;


#if 0
  printf("iclust : ");
  for (int i = 0; i < iclust_shape.Dims(0); ++i) {
    printf("[");
    for (int j = 0; j < iclust_shape.Dims(1); ++j) {
      printf("%d, ", iclust_data[INDEX_2D(iclust_shape, i, j)]);
    }
    printf("], ");
  }
  printf("\n");
  printf("oclust : ");
  for (int i = 0; i < oclust_shape.Dims(0); ++i) {
    printf("[");
    for (int j = 0; j < oclust_shape.Dims(1); ++j) {
      printf("%d, ", oclust_data[i * oclust_shape.Dims(1) + j]);
    }
    printf("], ");
  }
  printf("\n");
#endif

	int count = output_shape.FlatSize();
	for (int i = 0; i < count; ++i) {
		output_data[i] = 0.5;
	}

	return kTfLiteOk;
}

#undef   DIMS_1
#undef   DIMS_2
#undef   DIMS_3
#undef   DIMS_4
#undef   DIMS_5
#undef   DIMS_6
#undef INDEX_2D
#undef INDEX_3D
#undef INDEX_4D
#undef INDEX_5D
#undef INDEX_6D

} // namespace dragunov

TfLiteRegistration *Register_DRAGUNOV()
{
	static TfLiteRegistration r = {dragunov::Init, dragunov::Free, dragunov::Prepare, dragunov::Eval};
	return &r;
}

} // namespace ops
} // namespace tflite
