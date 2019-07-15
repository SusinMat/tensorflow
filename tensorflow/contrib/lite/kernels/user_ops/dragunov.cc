#include "tensorflow/contrib/lite/kernels/eigen_support.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/multithreaded_conv.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/padding.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/user_ops/user_ops.h"
#include "tensorflow/contrib/lite/model.h"

#include "tensorflow/contrib/lite/profiler.h"
#include <chrono>

#include "flatbuffers/flexbuffers.h" // TF:flatbuffers

namespace tflite {
namespace ops {
namespace dragunov {

#define     DIM1(shape) shape.Dims(1)
#define     DIM2(shape) shape.Dims(2)
#define     DIM3(shape) shape.Dims(3)
#define     DIM4(shape) shape.Dims(4)
#define     DIM5(shape) shape.Dims(5)
#define     DIM6(shape) shape.Dims(6)
#define INDEX_2D(shape, i, j)             (i * DIM1(shape) + j)
#define INDEX_3D(shape, i, j, k)          (i * DIM1(shape) * DIM2(shape) + j * DIM2(shape) + k)
#define INDEX_4D(shape, i, j, k, l)       (i * DIM1(shape) * DIM2(shape) * DIM3(shape) + j * DIM2(shape) * DIM3(shape) + k * DIM3(shape) + l)
// INDEX_4D can be replaced with SubscriptToIndex from tensorflow/contrib/lite/kernels/internal/common.h
//                       or with Offset           from tensorflow/contrib/lite/kernels/internal/types.h
#define INDEX_5D(shape, i, j, k, l, m)    (i * DIM1(shape) * DIM2(shape) * DIM3(shape) * DIM4(shape) + j * DIM2(shape) * DIM3(shape) * DIM4(shape) + k * DIM3(shape) * DIM4(shape) + l * DIM4(shape) + m)
#define INDEX_6D(shape, i, j, k, l, m, n) (i * DIM1(shape) * DIM2(shape) * DIM3(shape) * DIM4(shape) * DIM5(shape) + j * DIM2(shape) * DIM3(shape) * DIM4(shape) * DIM5(shape) + k * DIM3(shape) * DIM4(shape) * DIM5(shape) + l * DIM4(shape) * DIM5(shape) + m * DIM5(shape) + n)

struct TfLiteDragunovParams
{
  int stride_h;
  int stride_w;
  TfLitePadding padding_type;
  TfLitePaddingValues padding_values;
  TfLiteFusedActivation activation;

  float *sliced_input;
  float *C_filters;
  float *C_outputs;
  float *Z_outputs;
  float *Z_filters;
  float *F_outputs;
  float *F_filters;
};

const int dilation_height_factor = 1;
const int dilation_width_factor = 1;

constexpr int INPUT_TENSOR = 0;
constexpr int C_TENSOR = 1;
constexpr int Z_TENSOR = 2;
constexpr int F_TENSOR = 3;
constexpr int ICLUSTERS_TENSOR = 4;
constexpr int OCLUSTERS_TENSOR = 5;
constexpr int BIAS_TENSOR = 6;
constexpr int OUTPUT_TENSOR = 0;

inline PaddingType RuntimePaddingType(TfLitePadding padding_type)
{
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

void *Init(TfLiteContext *context, const char *buffer, size_t length)
{
  struct TfLiteDragunovParams *data = new struct TfLiteDragunovParams;

  const uint8_t *buffer_t = reinterpret_cast<const uint8_t *>(buffer);

  const flexbuffers::Map &m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  data->stride_h = m["stride_h"].AsInt64();
  data->stride_w = m["stride_w"].AsInt64();
  data->padding_type = (TfLitePadding)m["padding"].AsInt64();
  data->activation = TfLiteFusedActivation(m["fused_activation_function"].AsInt64());

  return data;
}

void Free(TfLiteContext *context, void *buffer)
{
  free(reinterpret_cast<struct TfLiteDragunovParams *>(buffer)->sliced_input);
  free(reinterpret_cast<struct TfLiteDragunovParams *>(buffer)->C_filters);
  free(reinterpret_cast<struct TfLiteDragunovParams *>(buffer)->C_outputs);
  free(reinterpret_cast<struct TfLiteDragunovParams *>(buffer)->Z_filters);
  free(reinterpret_cast<struct TfLiteDragunovParams *>(buffer)->Z_outputs);
  free(reinterpret_cast<struct TfLiteDragunovParams *>(buffer)->F_filters);
  free(reinterpret_cast<struct TfLiteDragunovParams *>(buffer)->F_outputs);

  delete reinterpret_cast<struct TfLiteDragunovParams *>(buffer);

  return;
}

void TransposeFloatArray(float *input_data, float *output_data, int rows, int cols)
{
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const float in_value = input_data[i * cols + j];
      output_data[j * rows + i] = in_value;
    }
  }
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node)
{
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
	const TfLiteTensor *iclusters = GetInput(context, node, ICLUSTERS_TENSOR);
	const TfLiteTensor *oclusters = GetInput(context, node, OCLUSTERS_TENSOR);
	const TfLiteTensor *c_filter = GetInput(context, node, C_TENSOR);
	const TfLiteTensor *z_filter = GetInput(context, node, Z_TENSOR);
	const TfLiteTensor *f_filter = GetInput(context, node, F_TENSOR);
	TfLiteTensor *output = GetOutput(context, node, OUTPUT_TENSOR);

	const float *input_data = GetTensorData<float>(input);
	const float *c_filter_data = GetTensorData<float>(c_filter);
	const float *z_filter_data = GetTensorData<float>(z_filter);
	const float *f_filter_data = GetTensorData<float>(f_filter);
	const int *iclusters_data = GetTensorData<int>(iclusters);
	const int *oclusters_data = GetTensorData<int>(oclusters);
	float *output_data = GetTensorData<float>(output);

  const RuntimeShape input_shape = GetTensorShape(input);
  const RuntimeShape output_shape = GetTensorShape(output);
  const RuntimeShape iclusters_shape = GetTensorShape(iclusters);
  const RuntimeShape oclusters_shape = GetTensorShape(oclusters);
  const RuntimeShape c_filter_shape = GetTensorShape(c_filter);
  const RuntimeShape z_filter_shape = GetTensorShape(z_filter);
  const RuntimeShape f_filter_shape = GetTensorShape(f_filter);

  const int iclust              = iclusters_shape.Dims(0);
  const int oclust              = oclusters_shape.Dims(0);
  const int cluster_pairs       = iclust * oclust;
  const int iclust_size         = c_filter_shape.Dims(0);
  const int iclust_reduced_size = c_filter_shape.Dims(1);
  const int oclust_size         = f_filter_shape.Dims(0);
  const int oclust_reduced_size = f_filter_shape.Dims(1);
  const int input_height        = input_shape.Dims(1);
  const int input_width         = input_shape.Dims(2);
  const int filter_height       = z_filter_shape.Dims(1);
  const int filter_width        = z_filter_shape.Dims(2);
  const int output_depth        = oclust * oclust_size;
  int output_height             = -1;
  int output_width              = -1;

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  if (params->padding_type == kTfLitePaddingSame) {
    output_height = (int)ceilf(float(input_height) / float(params->stride_h));
    output_width  = (int)ceilf(float(input_width) / float(params->stride_w));
  } else if (params->padding_type == kTfLitePaddingValid) {
    output_height = (int)ceilf(float(input_height - filter_height + 1) / float(params->stride_h));
    output_width  = (int)ceilf(float(input_width - filter_width + 1) / float(params->stride_w));
  }
  params->padding_values.height = ComputePadding(params->stride_h, dilation_height_factor, input_height, filter_height, output_height);
  params->padding_values.width = ComputePadding(params->stride_w, dilation_width_factor, input_width, filter_width, output_width);
	TfLiteIntArray *output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = 1;
  output_size->data[1] = output_height;
  output_size->data[2] = output_width;
  output_size->data[3] = output_depth;
  TfLiteStatus retval = context->ResizeTensor(context, output, output_size);

	const int output_flat_size = output_height * output_width * output_depth;

  const RuntimeShape C_input_shape({1, input_height, input_width, iclust_size});
  const RuntimeShape C_filter_shape({iclust_reduced_size, 1, 1, iclust_size});
  const RuntimeShape C_filter_shape_hwcn({1, 1, iclust_size, iclust_reduced_size});
  const RuntimeShape C_output_shape({1, input_height, input_width, iclust_reduced_size});

  const RuntimeShape Z_filter_shape({oclust_reduced_size, filter_height, filter_width, iclust_reduced_size});
  const RuntimeShape Z_filter_shape_hwcn({filter_height, filter_width, iclust_reduced_size, oclust_reduced_size}); // hwcn
  const RuntimeShape Z_output_shape({1, output_height, output_width, oclust_reduced_size});

  const RuntimeShape F_filter_shape({oclust_size, 1, 1, oclust_reduced_size});
  const RuntimeShape F_filter_shape_hwcn({1, 1, oclust_reduced_size, oclust_size});
  const RuntimeShape F_output_shape({1, output_height, output_width, oclust_size});

  const int C_input_flat_size = C_input_shape.FlatSize();
  const int C_filter_flat_size = C_filter_shape.FlatSize();
  const int C_output_flat_size = C_output_shape.FlatSize();

  const int Z_filter_flat_size = Z_filter_shape.FlatSize();
  const int Z_output_flat_size = Z_output_shape.FlatSize();

  const int F_filter_flat_size = F_filter_shape.FlatSize();
  const int F_output_flat_size = F_output_shape.FlatSize();

  params->sliced_input = (float *)calloc(iclust * C_input_flat_size, sizeof(float));
  params->C_filters = (float *)calloc(cluster_pairs * C_filter_flat_size, sizeof(float));
  params->C_outputs = (float *)calloc(cluster_pairs * C_output_flat_size, sizeof(float));
  params->Z_outputs = (float *)calloc(cluster_pairs * Z_output_flat_size, sizeof(float));
  params->Z_filters = (float *)calloc(cluster_pairs * Z_filter_flat_size, sizeof(float));
  params->F_outputs = (float *)calloc(cluster_pairs * F_output_flat_size, sizeof(float));
  params->F_filters = (float *)calloc(cluster_pairs * F_filter_flat_size, sizeof(float));

  for (int ic = 0; ic < iclust; ++ic) {
    for (int oc = 0; oc < oclust; ++oc) {
      for (int c = 0; c < iclust_size; ++c) {
        for (int f = 0; f < iclust_reduced_size; ++f) {
          (params->C_filters + (ic * oclust + oc) * C_filter_flat_size)[INDEX_4D(C_filter_shape_hwcn, 0, 0, c, f)] = c_filter_data[INDEX_4D(c_filter_shape, c, f, ic, oc)];
        }
      }
    }
  }

  for (int ic = 0; ic < iclust; ++ic) {
    for (int oc = 0; oc < oclust; ++oc) {
      for (int i = 0; i < filter_height; ++i) {
        for (int j = 0; j < filter_width; ++j) {
          for (int c = 0; c < iclust_reduced_size; ++c) {
            for (int f = 0; f < oclust_reduced_size; ++f) {
              (params->Z_filters + (ic * oclust + oc) * Z_filter_flat_size)[INDEX_4D(Z_filter_shape_hwcn, i, j, c, f)] = z_filter_data[INDEX_6D(z_filter_shape, f, i, j, c, ic, oc)];
            }
          }
        }
      }
    }
  }

  for (int ic = 0; ic < iclust; ++ic) {
    for (int oc = 0; oc < oclust; ++oc) {
      for (int c = 0; c < oclust_reduced_size; ++c) {
        for (int f = 0; f < oclust_size; ++f) {
          (params->F_filters + (ic * oclust + oc) * F_filter_flat_size)[INDEX_4D(F_filter_shape_hwcn, 0, 0, c, f)] = f_filter_data[INDEX_4D(f_filter_shape, f, c, ic, oc)];
        }
      }
    }
  }

  return retval;
}

std::chrono::system_clock::time_point CurrentTime()
{
  return std::chrono::system_clock::now();
}

inline std::chrono::milliseconds::rep TimePointInMilliseconds(std::chrono::system_clock::time_point timepoint)
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(timepoint.time_since_epoch()).count();
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node)
{
  std::chrono::system_clock::time_point start, stop;
  Profiler::KernelExecutionMeasure slicing, phase_C, phase_Z, phase_F, sum, bias_relu;
  struct TfLiteDragunovParams *params = reinterpret_cast<TfLiteDragunovParams *>(node->user_data);
	const TfLiteTensor *input = GetInput(context, node, INPUT_TENSOR);
	const TfLiteTensor *iclusters = GetInput(context, node, ICLUSTERS_TENSOR);
	const TfLiteTensor *oclusters = GetInput(context, node, OCLUSTERS_TENSOR);
	const TfLiteTensor *c_filter = GetInput(context, node, C_TENSOR);
	const TfLiteTensor *z_filter = GetInput(context, node, Z_TENSOR);
	const TfLiteTensor *f_filter = GetInput(context, node, F_TENSOR);
	const TfLiteTensor *bias = GetInput(context, node, BIAS_TENSOR);
	TfLiteTensor *output = GetOutput(context, node, OUTPUT_TENSOR);

	const float *input_data = GetTensorData<float>(input);
	const float *c_filter_data = GetTensorData<float>(c_filter);
	const float *z_filter_data = GetTensorData<float>(z_filter);
	const float *f_filter_data = GetTensorData<float>(f_filter);
	const int *iclusters_data = GetTensorData<int>(iclusters);
	const int *oclusters_data = GetTensorData<int>(oclusters);
	const float *bias_data = GetTensorData<float>(bias);
	float *output_data = GetTensorData<float>(output);

  const RuntimeShape input_shape = GetTensorShape(input);
  const RuntimeShape output_shape = GetTensorShape(output);
  const RuntimeShape iclusters_shape = GetTensorShape(iclusters);
  const RuntimeShape oclusters_shape = GetTensorShape(oclusters);
  const RuntimeShape c_filter_shape = GetTensorShape(c_filter);
  const RuntimeShape z_filter_shape = GetTensorShape(z_filter);
  const RuntimeShape f_filter_shape = GetTensorShape(f_filter);
  const RuntimeShape bias_shape = GetTensorShape(bias);
  const RuntimeShape null_tensor_shape;
  float *null_tensor_data = nullptr;

  const int iclust              = iclusters_shape.Dims(0);
  const int oclust              = oclusters_shape.Dims(0);
  const int cluster_pairs       = iclust * oclust;
  const int iclust_size         = c_filter_shape.Dims(0);
  const int iclust_reduced_size = c_filter_shape.Dims(1);
  const int oclust_size         = f_filter_shape.Dims(0);
  const int oclust_reduced_size = f_filter_shape.Dims(1);
  const int input_height        = input_shape.Dims(1);
  const int input_width         = input_shape.Dims(2);
  const int filter_height       = z_filter_shape.Dims(1);
  const int filter_width        = z_filter_shape.Dims(2);
  const int output_height       = output_shape.Dims(1);
  const int output_width        = output_shape.Dims(2);
  const int output_depth        = output_shape.Dims(3);
	const int output_flat_size    = output_shape.FlatSize();

  const RuntimeShape C_input_shape({1, input_height, input_width, iclust_size});
  const RuntimeShape C_filter_shape({iclust_reduced_size, 1, 1, iclust_size});
  const RuntimeShape C_filter_shape_hwcn({1, 1, iclust_size, iclust_reduced_size});
  const RuntimeShape C_output_shape({1, input_height, input_width, iclust_reduced_size});

  const RuntimeShape Z_filter_shape({oclust_reduced_size, filter_height, filter_width, iclust_reduced_size});
  const RuntimeShape Z_filter_shape_hwcn({filter_height, filter_width, iclust_reduced_size, oclust_reduced_size}); // hwcn
  const RuntimeShape Z_output_shape({1, output_height, output_width, oclust_reduced_size});

  const RuntimeShape F_filter_shape({oclust_size, 1, 1, oclust_reduced_size});
  const RuntimeShape F_filter_shape_hwcn({1, 1, oclust_reduced_size, oclust_size});
  const RuntimeShape F_output_shape({1, output_height, output_width, oclust_size});

  const int C_input_flat_size = C_input_shape.FlatSize();
  const int C_filter_flat_size = C_filter_shape.FlatSize();
  const int C_output_flat_size = C_output_shape.FlatSize();

  const int Z_filter_flat_size = Z_filter_shape.FlatSize();
  const int Z_output_flat_size = Z_output_shape.FlatSize();

  const int F_filter_flat_size = F_filter_shape.FlatSize();
  const int F_output_flat_size = F_output_shape.FlatSize();

  float *sliced_input = params->sliced_input;
  const float *C_filters = params->C_filters;
  float *C_outputs = params->C_outputs;
  float *Z_outputs = params->Z_outputs;
  const float *Z_filters = params->Z_filters;
  float *F_outputs = params->F_outputs;
  const float *F_filters = params->F_filters;

  ConvParams op_params;
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min, &output_activation_max);

#if 0
  printf("iclust = %d ; oclust = %d\n", iclust, oclust);

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
#endif

  // Phase C
  phase_C.name = "Dragunov_Phase_C";
  start = CurrentTime();
  phase_C.start_clock = TimePointInMilliseconds(start);

  // im2col not required yet, unless (stride_h, stride_w) != (1, 1)
  op_params.padding_type = PaddingType::kNone;
  op_params.padding_values.height = 0;
  op_params.padding_values.width = 0;
  op_params.stride_height = 1;
  op_params.stride_width = 1;
  op_params.dilation_height_factor = dilation_height_factor;
  op_params.dilation_width_factor = dilation_width_factor;

  for (int ic = 0; ic < iclust; ++ic) {
    for (int i = 0; i < input_height; ++i) {
      for (int j = 0; j < input_width; ++j) {
        for (int k = 0; k < iclust_size; ++k) {
          int source_iclust = iclusters_data[INDEX_2D(iclusters_shape, ic, k)];
          (sliced_input + ic * C_input_flat_size)[INDEX_4D(C_input_shape, 0, i, j, k)] = input_data[INDEX_4D(input_shape, 0, i, j, source_iclust)];
        }
      }
    }
  }

  for (int ic = 0; ic < iclust; ++ic) {
    for (int oc = 0; oc < oclust; ++oc) {
      multithreaded_ops::Conv(
          *eigen_support::GetThreadPoolDevice(context),
          op_params,
          C_input_shape, (const float *)(sliced_input + ic * C_input_flat_size),
          C_filter_shape, (const float *)(C_filters + (ic * oclust + oc) * C_filter_flat_size),
          null_tensor_shape, null_tensor_data,
          C_output_shape, (float *)(C_outputs + (ic * oclust + oc) * C_output_flat_size),
          null_tensor_shape, null_tensor_data);
    }
  }

  stop = CurrentTime();
  phase_C.stop_clock = TimePointInMilliseconds(stop);
  phase_C.execution_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  Profiler::get().addKernelMeasure(phase_C);

  // Phase Z
  phase_Z.name = "Dragunov_Phase_Z";
  start = CurrentTime();
  phase_Z.start_clock = TimePointInMilliseconds(start);

  op_params.padding_type = RuntimePaddingType(params->padding_type);
  op_params.padding_values.height = params->padding_values.height;
  op_params.padding_values.width = params->padding_values.width;
  op_params.stride_height = params->stride_h;
  op_params.stride_width = params->stride_w;
  op_params.dilation_height_factor = dilation_height_factor;
  op_params.dilation_width_factor = dilation_width_factor;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  for (int ic = 0; ic < iclust; ++ic) {
    for (int oc = 0; oc < oclust; ++oc) {
      multithreaded_ops::Conv(
          *eigen_support::GetThreadPoolDevice(context),
          op_params,
          C_output_shape, (const float *)(C_outputs + (ic * oclust + oc) * C_output_flat_size),
          Z_filter_shape, (const float *)(Z_filters + (ic * oclust + oc) * Z_filter_flat_size),
          null_tensor_shape, null_tensor_data,
          Z_output_shape, (float *)(Z_outputs + (ic * oclust + oc) * Z_output_flat_size),
          null_tensor_shape, null_tensor_data);
    }
  }

  stop = CurrentTime();
  phase_Z.stop_clock = TimePointInMilliseconds(stop);
  phase_Z.execution_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  Profiler::get().addKernelMeasure(phase_Z);

  // Phase F
  phase_F.name = "Dragunov_Phase_F";
  start = CurrentTime();
  phase_F.start_clock = TimePointInMilliseconds(start);

  op_params.padding_type = PaddingType::kNone;
  op_params.padding_values.height = 0;
  op_params.padding_values.width = 0;
  op_params.stride_height = 1;
  op_params.stride_width = 1;
  op_params.dilation_height_factor = dilation_height_factor;
  op_params.dilation_width_factor = dilation_width_factor;

  for (int ic = 0; ic < iclust; ++ic) {
    for (int oc = 0; oc < oclust; ++oc) {
      multithreaded_ops::Conv(
          *eigen_support::GetThreadPoolDevice(context),
          op_params,
          Z_output_shape, (const float *)(Z_outputs + (ic * oclust + oc) * Z_output_flat_size),
          F_filter_shape, (const float *)(F_filters + (ic * oclust + oc) * F_filter_flat_size),
          null_tensor_shape, null_tensor_data,
          F_output_shape, (float *)(F_outputs + (ic * oclust + oc) * F_output_flat_size),
          null_tensor_shape, null_tensor_data);
    }
  }

	for (int i = 0; i < output_flat_size; ++i) {
		output_data[i] = 0.0f;
	}

  for (int ic = 0; ic < iclust; ++ic) {
    for (int oc = 0; oc < oclust; ++oc) {
      for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
          for (int c = 0; c < oclust_size; ++c) {
            int dest_channel = oclusters_data[INDEX_2D(oclusters_shape, oc, c)];
             output_data[INDEX_4D(output_shape, 0, i, j, dest_channel)] += (F_outputs + (ic * oclust + oc) * F_output_flat_size)[INDEX_4D(F_output_shape, 0, i, j, c)];
          }
        }
      }
    }
  }

  optimized_ops::AddBiasAndEvalActivationFunction(
      output_activation_min, output_activation_max,
      bias_shape, bias_data,
      output_shape, output_data);

  stop = CurrentTime();
  phase_F.stop_clock = TimePointInMilliseconds(stop);
  phase_F.execution_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  Profiler::get().addKernelMeasure(phase_F);

	return kTfLiteOk;
}

#undef     DIM1
#undef     DIM2
#undef     DIM3
#undef     DIM4
#undef     DIM5
#undef     DIM6
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
