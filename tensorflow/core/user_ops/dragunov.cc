#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

Status DragunovShape(shape_inference::InferenceContext* c) {
  int padding, stride_h, stride_w,
      out_height, out_width, in_height, in_width, filter_height, filter_width;
  TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));
  TF_RETURN_IF_ERROR(c->GetAttr("stride_h", &stride_h));
  TF_RETURN_IF_ERROR(c->GetAttr("stride_w", &stride_w));

  shape_inference::DimensionHandle output_channels;
  shape_inference::ShapeHandle dims_f;
  shape_inference::ShapeHandle input_shape;
  shape_inference::ShapeHandle output_shape;
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
    .Input("input_clusters: int32")
    .Input("output_clusters: int32")
    .Output("output: float")
    .Attr("stride_h: int = 1")
    .Attr("stride_w: int = 1")
    .Attr("padding: int = 0") // TODO: enum class Padding: int8_t { UNKNOWN = 0, SAME, VALID };
    .SetShapeFn(DragunovShape);


class DragunovOp : public tensorflow::OpKernel {
 public:
  explicit DragunovOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    (void)0;
  }
};

REGISTER_KERNEL_BUILDER(Name("Dragunov").Device(tensorflow::DEVICE_CPU), DragunovOp);
} // namespace tensorflow
