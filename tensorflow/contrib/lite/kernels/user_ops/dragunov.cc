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
  printf("Padding type: %d\n", params->padding);

	TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
	TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

	const TfLiteTensor *input = GetInput(context, node, INPUT_TENSOR);
	const TfLiteTensor *f_filter = GetInput(context, node, F_TENSOR);
	TfLiteTensor *output = GetOutput(context, node, OUTPUT_TENSOR);

  TF_LITE_ENSURE_EQ(context, input->type, output->type);

#if 0
	const TfLiteTensor *c_filter = GetInput(context, node, C_TENSOR);
	const TfLiteTensor *z_filter = GetInput(context, node, Z_TENSOR);
	const int input_dims = NumDimensions(input);
  printf("input dims: ");
  for (int i = 0; i < input_dims; ++i)
    printf("%d ", input->dims->data[i]);
  printf("\n");

	const int c_dims = NumDimensions(c_filter);
  printf("c_filter dims: ");
  for (int i = 0; i < c_dims; ++i)
    printf("%d ", c_filter->dims->data[i]);
  printf("\n");

	const int z_dims = NumDimensions(z_filter);
  printf("z_filter dims: ");
  for (int i = 0; i < z_dims; ++i)
    printf("%d ", z_filter->dims->data[i]);
  printf("\n");

	const int f_dims = NumDimensions(f_filter);
  printf("f_filter dims: ");
  for (int i = 0; i < f_dims; ++i)
    printf("%d ", f_filter->dims->data[i]);
  printf("\n");
#endif

	TfLiteIntArray *output_size = TfLiteIntArrayCreate(4);

  output_size->data[0] = 1;
  output_size->data[1] = input->dims->data[1];
  output_size->data[2] = input->dims->data[2];
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
