#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
#if 0
Status ConvShape(shape_inference::InferenceContext* c) {
  string data_format_str, filter_format_str;
  if (!c->GetAttr("data_format", &data_format_str).ok()) {
    data_format_str = "NHWC";
  }
  if (!c->GetAttr("filter_format", &filter_format_str).ok()) {
    filter_format_str = "HWIO";
  }

  TensorFormat data_format;
  if (!FormatFromString(data_format_str, &data_format)) {
    return errors::InvalidArgument("Invalid data format string: ",
                                   data_format_str);
  }
  FilterTensorFormat filter_format;
  if (!FilterFormatFromString(filter_format_str, &filter_format)) {
    return errors::InvalidArgument("Invalid filter format string: ",
                                   filter_format_str);
  }

  constexpr int num_spatial_dims = 2;
  const int rank = GetTensorDimsFromSpatialDims(num_spatial_dims, data_format);
  ShapeHandle conv_input_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), rank, &conv_input_shape));
  TF_RETURN_IF_ERROR(CheckFormatConstraintsOnShape(
      data_format, conv_input_shape, "conv_input", c));

  // The filter rank should match the input (4 for NCHW, 5 for NCHW_VECT_C).
  ShapeHandle filter_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), rank, &filter_shape));
  TF_RETURN_IF_ERROR(
      CheckFormatConstraintsOnShape(data_format, filter_shape, "filter", c));

  std::vector<int32> dilations;
  TF_RETURN_IF_ERROR(c->GetAttr("dilations", &dilations));

  if (dilations.size() != 4) {
    return errors::InvalidArgument(
        "Conv2D requires the dilation attribute to contain 4 values, but got: ",
        dilations.size());
  }

  std::vector<int32> strides;
  TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));

  // strides.size() should be 4 (NCHW) even if the input is 5 (NCHW_VECT_C).
  if (strides.size() != 4) {
    return errors::InvalidArgument("Conv2D on data format ", data_format_str,
                                   " requires the stride attribute to contain"
                                   " 4 values, but got: ",
                                   strides.size());
  }

  const int32 stride_rows = GetTensorDim(strides, data_format, 'H');
  const int32 stride_cols = GetTensorDim(strides, data_format, 'W');
  const int32 dilation_rows = GetTensorDim(dilations, data_format, 'H');
  const int32 dilation_cols = GetTensorDim(dilations, data_format, 'W');

  DimensionHandle batch_size_dim;
  DimensionHandle input_depth_dim;
  gtl::InlinedVector<DimensionHandle, 2> input_spatial_dims(2);
  TF_RETURN_IF_ERROR(DimensionsFromShape(
      conv_input_shape, data_format, &batch_size_dim,
      absl::MakeSpan(input_spatial_dims), &input_depth_dim, c));

  DimensionHandle output_depth_dim = c->Dim(
      filter_shape, GetFilterDimIndex<num_spatial_dims>(filter_format, 'O'));
  DimensionHandle filter_rows_dim = c->Dim(
      filter_shape, GetFilterDimIndex<num_spatial_dims>(filter_format, 'H'));
  DimensionHandle filter_cols_dim = c->Dim(
      filter_shape, GetFilterDimIndex<num_spatial_dims>(filter_format, 'W'));
  DimensionHandle filter_input_depth_dim;
  if (filter_format == FORMAT_OIHW_VECT_I) {
    TF_RETURN_IF_ERROR(c->Multiply(
        c->Dim(filter_shape,
               GetFilterDimIndex<num_spatial_dims>(filter_format, 'I')),
        c->Dim(filter_shape,
               GetFilterTensorInnerInputChannelsDimIndex(rank, filter_format)),
        &filter_input_depth_dim));
  } else {
    filter_input_depth_dim = c->Dim(
        filter_shape, GetFilterDimIndex<num_spatial_dims>(filter_format, 'I'));
  }

  // Check that the input tensor and the filter tensor agree on the input
  // channel count.
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(
      c->Merge(input_depth_dim, filter_input_depth_dim, &unused));

  Padding padding;
  TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

  DimensionHandle output_rows, output_cols;
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDimsV2(
      c, input_spatial_dims[0], filter_rows_dim, dilation_rows, stride_rows,
      padding, &output_rows));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDimsV2(
      c, input_spatial_dims[1], filter_cols_dim, dilation_cols, stride_cols,
      padding, &output_cols));

  ShapeHandle output_shape;
  TF_RETURN_IF_ERROR(
      ShapeFromDimensions(batch_size_dim, {output_rows, output_cols},
                          output_depth_dim, data_format, c, &output_shape));
  c->set_output(0, output_shape);
  return Status::OK();
}
#endif

Status DragunovShape(shape_inference::InferenceContext* c) {
  int padding, stride_h, stride_w,
      out_height, out_width, in_height, in_width, filter_height, filter_width;
  TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));
  TF_RETURN_IF_ERROR(c->GetAttr("stride_h", &stride_h));
  TF_RETURN_IF_ERROR(c->GetAttr("stride_w", &stride_w));
  // For the SAME(1) padding, the output height and width are computed as:
  //
  // out_height = ceil(float(in_height) / float(strides[1]))
  // out_width  = ceil(float(in_width) / float(strides[2]))
  //
  // And
  //
  // For the VALID(2) padding, the output height and width are computed as:
  //
  // out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
  // out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))

  shape_inference::DimensionHandle output_channels;
  shape_inference::ShapeHandle dims_f;
#if 1
  shape_inference::ShapeHandle input_shape;
  shape_inference::ShapeHandle dims_z;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 6, &dims_z));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
  in_height = c->Value(c->Dim(input_shape, 1));
  in_width = c->Value(c->Dim(input_shape, 2));
  filter_height = c->Value(c->Dim(dims_z, 1));
  filter_width = c->Value(c->Dim(dims_z, 2));
  if (padding == 1) {
    out_height = (int)ceilf(float(in_height) / float(stride_h));
    out_width  = (int)ceilf(float(in_width) / float(stride_w));
  } else if (padding == 2) {
    out_height = (int)ceilf(float(in_height - filter_height + 1) / float(stride_h));
    out_width  = (int)ceilf(float(in_width - filter_width + 1) / float(stride_w));
  } else {
    out_height = -1;
    out_width = -1;
  }
#endif
  shape_inference::ShapeHandle output_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 4, &dims_f));
  TF_RETURN_IF_ERROR(c->Multiply(c->Dim(dims_f, 0), c->Dim(dims_f, 3), &output_channels));

  output_shape = c->MakeShape({c->MakeDim(1), c->MakeDim(out_height), c->MakeDim(out_width), output_channels});
  c->set_output(0, output_shape);
  return Status::OK();
}

REGISTER_OP("Dragunov")
    .Input("input: float")
    .Input("filter_c: float")
    .Input("filter_z: float")
    .Input("filter_f: float")
    .Output("output: float")
    .Attr("stride_h: int")
    .Attr("stride_w: int")
    .Attr("padding: int") // TODO: enum class Padding: int8_t { UNKNOWN = 0, SAME, VALID };
    .SetShapeFn(DragunovShape);


class DragunovOp : public tensorflow::OpKernel {
 public:
  explicit DragunovOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    // Output a scalar string.
    tensorflow::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, tensorflow::TensorShape(), &output_tensor));
    using tensorflow::string;
    auto output = output_tensor->template scalar<string>();

    output() = "0! == 1";
  }
};

REGISTER_KERNEL_BUILDER(Name("Dragunov").Device(tensorflow::DEVICE_CPU), DragunovOp);
} // namespace tensorflow
