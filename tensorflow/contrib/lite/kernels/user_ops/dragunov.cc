#include "tensorflow/contrib/lite/kernels/user_ops/user_ops.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"

namespace tflite {
namespace ops {
namespace dragunov {

constexpr int input_tensor = 0;
constexpr int output_tensor = 0;

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node)
{
	TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
	TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

	const TfLiteTensor *input = GetInput(context, node, input_tensor);
	TfLiteTensor *output = GetOutput(context, node, output_tensor);

	if (input->type == kTfLiteFloat32 || output->type == kTfLiteFloat32) {
		TF_LITE_ENSURE_EQ(context, input->type, output->type);
	}

	TfLiteIntArray *output_size = TfLiteIntArrayCopy(input->dims);

	return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node)
{
	const TfLiteTensor *input = GetInput(context, node, input_tensor);
	TfLiteTensor *output = GetOutput(context, node, output_tensor);

	const float *input_data = GetTensorData<float>(input);
	float *output_data = GetTensorData<float>(output);
	size_t count = 1;
	int num_dims = NumDimensions(input);
	for (size_t i = 0; i < num_dims; ++i) {
		count *= input->dims->data[i];
	}
	for (size_t i = 0; i < count; ++i) {
		output_data[i] = input_data[i] + 1.0;
	}

	return kTfLiteOk;
}

} // namespace dragunov

TfLiteRegistration *Register_DRAGUNOV()
{
	static TfLiteRegistration r = {nullptr, nullptr, dragunov::Prepare, dragunov::Eval};
	return &r;
}

} // namespace ops
} // namespace tflite
