#include "tensorflow/contrib/lite/kernels/user_ops/user_ops.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"

#include "flatbuffers/flexbuffers.h" // TF:flatbuffers

namespace tflite {
namespace ops {
namespace dragunov {

struct TfLiteDragunovParams {
  int stride_h;
  int stride_w;
  int padding;
};

constexpr int INPUT_TENSOR = 0;
constexpr int OUTPUT_TENSOR = 0;
constexpr int C_TENSOR = 1;
constexpr int Z_TENSOR = 2;
constexpr int F_TENSOR = 3;

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  struct TfLiteDragunovParams *data = new struct TfLiteDragunovParams;

  const uint8_t *buffer_t = reinterpret_cast<const uint8_t *>(buffer);

  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  data->stride_h = m["stride_h"].AsInt64();
  data->stride_w = m["stride_w"].AsInt64();
  data->padding = m["padding"].AsInt64();

  return data;
}

void Free(TfLiteContext *context, void *buffer) {
  delete reinterpret_cast<struct TfLiteDragunovParams *>(buffer);
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node)
{
  struct TfLiteDragunovParams *params = reinterpret_cast<TfLiteDragunovParams *>(node->user_data);

  printf("Strides [h, w]: [%d, %d]\n", params->stride_h, params->stride_w);
  printf("Padding type: ");

  switch(params->padding) {
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
      printf("ERROR(%d)\n", params->padding);
      break;
  }

	TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
	TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

	const TfLiteTensor *input = GetInput(context, node, INPUT_TENSOR);
	const TfLiteTensor *f_filter = GetInput(context, node, F_TENSOR);
	const TfLiteTensor *z_filter = GetInput(context, node, Z_TENSOR);
	TfLiteTensor *output = GetOutput(context, node, OUTPUT_TENSOR);

  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  int in_height = input->dims->data[1], in_width = input->dims->data[2];
  int filter_height = z_filter->dims->data[1], filter_width = z_filter->dims->data[2];
  int out_height = -1, out_width = -1;

  if (params->padding == 1) {
    out_height = (int)ceilf(float(in_height) / float(params->stride_h));
    out_width  = (int)ceilf(float(in_width) / float(params->stride_w));
  } else if (params->padding == 2) {
    out_height = (int)ceilf(float(in_height - filter_height + 1) / float(params->stride_h));
    out_width  = (int)ceilf(float(in_width - filter_width + 1) / float(params->stride_w));
  }

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
	const TfLiteTensor *input = GetInput(context, node, INPUT_TENSOR);
	TfLiteTensor *output = GetOutput(context, node, OUTPUT_TENSOR);

	const float *input_data = GetTensorData<float>(input);
	float *output_data = GetTensorData<float>(output);
  const RuntimeShape output_shape = GetTensorShape(output);
	int count = output_shape.FlatSize();
	int num_dims = output_shape.DimensionsCount();
  // printf("%d\n", output_shape.Dims(2));
  // printf("%d\n", count);

	for (int i = 0; i < count; ++i) {
		output_data[i] = 0.5;
	}

	return kTfLiteOk;
}

} // namespace dragunov

TfLiteRegistration *Register_DRAGUNOV()
{
	static TfLiteRegistration r = {dragunov::Init, dragunov::Free, dragunov::Prepare, dragunov::Eval};
	return &r;
}

} // namespace ops
} // namespace tflite
