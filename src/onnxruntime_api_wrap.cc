/* *
 * Copyright (c) 2020, ATP
 * All rights reserved.
 * MIT License
 */

#include "onnxruntime_api_wrap.h"

#include <cstdlib>
#include <numeric>
#include <string>

#include "onnxruntime_c_api.h"
#include "utilis/Defer.h"

namespace ONNX_NAMESPACE {
namespace inference {

ONNXTensorElementDataType TensorProtoDataTypeToONNXTensorElementDataType(
    const TensorProto_DataType elem_type) throw(std::invalid_argument) {
  ONNXTensorElementDataType type;
  switch (elem_type) {
  case TensorProto_DataType_UNDEFINED:
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    break;
  case TensorProto_DataType_BOOL:
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    break;

  case TensorProto_DataType_INT8:
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    break;
  case TensorProto_DataType_INT16:
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    break;
  case TensorProto_DataType_INT32:
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    break;
  case TensorProto_DataType_INT64:
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    break;

  case TensorProto_DataType_UINT8:
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    break;
  case TensorProto_DataType_UINT16:
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    break;
  case TensorProto_DataType_UINT32:
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    break;
  case TensorProto_DataType_UINT64:
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    break;

  case TensorProto_DataType_FLOAT:
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    break;
  case TensorProto_DataType_FLOAT16:
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    break;
  case TensorProto_DataType_BFLOAT16:
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    break;
  case TensorProto_DataType_DOUBLE:
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    break;
  case TensorProto_DataType_STRING:
    type = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    break;
  default:
    std::ostringstream err_msg;
    err_msg << "Tensor proto data type: '" << elem_type << "' convert failed.";
    throw std::invalid_argument(err_msg.str());
    break;
  }
  return type;
}

TensorProto_DataType ONNXTensorElementDataTypeToTensorProtoDataType(
    const ONNXTensorElementDataType elem_type) throw(std::invalid_argument) {
  TensorProto_DataType type;
  switch (elem_type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
    type = TensorProto_DataType_UNDEFINED;
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    type = TensorProto_DataType_BOOL;
    break;

  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    type = TensorProto_DataType_INT8;
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    type = TensorProto_DataType_INT16;
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    type = TensorProto_DataType_INT32;
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    type = TensorProto_DataType_INT64;
    break;

  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    type = TensorProto_DataType_UINT8;
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    type = TensorProto_DataType_UINT16;
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    type = TensorProto_DataType_UINT32;
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    type = TensorProto_DataType_UINT64;
    break;

  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    type = TensorProto_DataType_FLOAT;
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    type = TensorProto_DataType_FLOAT16;
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    type = TensorProto_DataType_BFLOAT16;
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    type = TensorProto_DataType_DOUBLE;
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
    type = TensorProto_DataType_STRING;
    break;

  default:
    std::ostringstream err_msg;
    err_msg << "Tensor element data type: '" << elem_type
            << "' convert failed.";
    throw std::invalid_argument(err_msg.str());
    break;
  }
  return type;
}

#define COPY_TENSOR_DATA(tensor, data_ptr, length, data_type)                  \
  tensor->data_type().resize(length);                                          \
  memcpy(tensor->data_type().data(), data_ptr,                                 \
         length * sizeof(std::remove_reference<decltype(                       \
                             tensor->data_type())>::type::value_type))

void SetTensorData(Tensor *tensor, void *data_ptr,
                   size_t length) throw(std::invalid_argument) {
  switch (tensor->elem_type()) {
  case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
    COPY_TENSOR_DATA(tensor, data_ptr, length, floats);
    break;
  }
  case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
    COPY_TENSOR_DATA(tensor, data_ptr, length, doubles);
    break;
  }
  case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
    COPY_TENSOR_DATA(tensor, data_ptr, length, int32s);
    break;
  }
  case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
    COPY_TENSOR_DATA(tensor, data_ptr, length, int64s);
    break;
  }
  case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {
    COPY_TENSOR_DATA(tensor, data_ptr, length, uint64s);
    break;
  }
  default:
    std::ostringstream err_msg;
    err_msg << "Tensor element type: '" << tensor->elem_type()
            << "' unsupported to set.";
    throw std::invalid_argument(err_msg.str());
    break;
  }
}

bool GetONNXTensorData(const Tensor &tensor, const void **data_ptr,
                       size_t *data_len) throw(std::invalid_argument) {
  *data_len = tensor.size_from_dim(0);
  if (tensor.is_raw_data()) {
    *data_ptr = tensor.raw().c_str();
    *data_len = tensor.raw().length();
  }
  switch (tensor.elem_type()) {
  case TensorProto_DataType_BOOL:
  case TensorProto_DataType_INT8:
  case TensorProto_DataType_INT16:
  case TensorProto_DataType_INT32:
  case TensorProto_DataType_UINT8:
  case TensorProto_DataType_UINT16:
    *data_ptr = tensor.data<int32_t>();
    *data_len *= sizeof(int32_t);
    break;
  case TensorProto_DataType_INT64:
    *data_ptr = tensor.data<int64_t>();
    *data_len *= sizeof(int64_t);
    break;
  case TensorProto_DataType_UINT32:
  case TensorProto_DataType_UINT64:
    *data_ptr = tensor.data<uint64_t>();
    *data_len *= sizeof(uint64_t);
    break;
  case TensorProto_DataType_FLOAT:
    *data_ptr = tensor.data<float>();
    *data_len *= sizeof(float);
    break;
  case TensorProto_DataType_DOUBLE:
    *data_ptr = tensor.data<double>();
    *data_len *= sizeof(double);
    break;
  default:
    std::ostringstream err_msg;
    err_msg << "Tensor element type: '" << tensor.elem_type()
            << "' unsupported to get.";
    throw std::invalid_argument(err_msg.str());
    break;
  }
  return true;
}

std::weak_ptr<OnnxApiWrapper> OnnxApiWrapper::wpWrapper_;
std::mutex OnnxApiWrapper::muxWrapper_;

const std::string
    OnnxApiWrapper::kOnnxruntimeLibName("onnxruntime_pybind11_state.so");
const std::string OnnxApiWrapper::kOrtGetApiBaseName("OrtGetApiBase");

OnnxApiWrapperPtr OnnxApiWrapper::GetInstance() throw(std::runtime_error) {
  // TODO(ATP): Here we don't use DCLP for the reason that,
  // 1. The cost of lock is acceptable, for this func is not called often.
  // 2. We don't use static member pattern to save the memory which shared
  //      library takes.
  std::lock_guard<std::mutex> guard(muxWrapper_);
  OnnxApiWrapperPtr ret = wpWrapper_.lock();
  if (ret == nullptr) {
    const LibraryLoader &loader = LibraryLoader::GetInstance();
    auto lib_handle_sptr = loader.LoadLibrary(kOnnxruntimeLibName);
    OrtGetApiBasePtr *func_api_base = loader.GetFuncPointer<OrtGetApiBasePtr>(
        lib_handle_sptr, kOrtGetApiBaseName);
    if (nullptr == func_api_base) {
      throw std::runtime_error("Onnxruntime api get failed!");
    }
    const OrtApi *p_ort = func_api_base()->GetApi(ORT_API_VERSION);
    ret = std::make_shared<OnnxApiWrapper>(lib_handle_sptr, p_ort);
    wpWrapper_ = ret;
  }
  return std::move(ret);
}

void OnnxApiWrapper::checkStatus(OrtStatus *status) const
    throw(std::runtime_error) {
  if (status != NULL) {
    const char *msg = pOrt_->GetErrorMessage(status);
    Utilis::DEFER([&] { pOrt_->ReleaseStatus(status); });
    fprintf(stderr, "%s\n", msg);
    throw std::runtime_error(msg);
  }
}

bool OnnxApiWrapper::InferModel(const ModelProto mp,
                                const std::vector<std::string> &input_names,
                                const std::vector<Tensor> &input_tensors,
                                const std::vector<std::string> &output_names,
                                std::vector<Tensor> *output_tensors) {
  // create execution environment
  OrtEnv *env;
  checkStatus(pOrt_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
  Utilis::DEFER([&] { pOrt_->ReleaseEnv(env); });

  // initialize session options if needed
  OrtSessionOptions *session_options;
  checkStatus(pOrt_->CreateSessionOptions(&session_options));
  Utilis::DEFER([&] { pOrt_->ReleaseSessionOptions(session_options); });

  pOrt_->SetIntraOpNumThreads(session_options, 1);
  pOrt_->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);

  // prepare model
  size_t model_size = mp.ByteSizeLong();
  std::vector<uint8_t> mp_buff(model_size);
  mp.SerializeToArray(mp_buff.data(), mp_buff.size());

  // create session
  OrtSession *sess;
  checkStatus(pOrt_->CreateSessionFromArray(env, mp_buff.data(), mp_buff.size(),
                                            session_options, &sess));
  Utilis::DEFER([&] { pOrt_->ReleaseSession(sess); });

  OrtMemoryInfo *memory_info;
  checkStatus(pOrt_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                         &memory_info));
  Utilis::DEFER([&] { pOrt_->ReleaseMemoryInfo(memory_info); });

  // prepare input data and info
  auto input_num = input_names.size();
  assert(input_num == input_tensors.size());

  auto ort_value_deleter = [&](OrtValue *val) { pOrt_->ReleaseValue(val); };

  std::vector<const char *> input_name_ptrs(input_num);
  std::vector<OrtValue *> inputs(input_num);
  std::vector<std::shared_ptr<OrtValue>> inputs_guard;
  for (size_t i = 0; i < input_num; ++i) {
    input_name_ptrs[i] = input_names[i].c_str();

    // transform input data holder
    void *data_ptr = nullptr;
    size_t data_len = 0;
    GetONNXTensorData(input_tensors[i], const_cast<const void **>(&data_ptr),
                      &data_len);

    OrtValue *input = nullptr;
    checkStatus(pOrt_->CreateTensorWithDataAsOrtValue(
        memory_info, data_ptr, data_len, input_tensors[i].sizes().data(),
        input_tensors[i].sizes().size(),
        TensorProtoDataTypeToONNXTensorElementDataType(
            static_cast<TensorProto_DataType>(input_tensors[i].elem_type())),
        &input));
    inputs[i] = input;
    inputs_guard.push_back(std::shared_ptr<OrtValue>(input, ort_value_deleter));
  }

  // prepare output data object
  auto output_num = output_names.size();
  std::vector<OrtValue *> outputs(output_num);
  std::vector<std::shared_ptr<OrtValue>> outputs_guard;

  output_tensors->resize(output_num);

  std::vector<const char *> output_name_ptrs(output_num);
  for (decltype(output_num) i = 0; i < output_num; ++i) {
    output_name_ptrs[i] = output_names[i].c_str();
  }

  // infer model
  checkStatus(pOrt_->Run(sess, nullptr, input_name_ptrs.data(), inputs.data(),
                         inputs.size(), output_name_ptrs.data(), output_num,
                         outputs.data()));

  for (decltype(output_num) i = 0; i < output_num; ++i) {
    outputs_guard.push_back(
        std::shared_ptr<OrtValue>(outputs[i], ort_value_deleter));
  }
  for (decltype(output_num) i = 0; i < output_num; ++i) {
    // get tensor type and shape info
    int is_tensor;
    checkStatus(pOrt_->IsTensor(outputs[i], &is_tensor));
    assert(is_tensor);
    OrtTensorTypeAndShapeInfo *type_shape_ptr;
    checkStatus(pOrt_->GetTensorTypeAndShape(outputs[i], &type_shape_ptr));
    Utilis::DEFER(
        [&] { pOrt_->ReleaseTensorTypeAndShapeInfo(type_shape_ptr); });

    // set tensor type
    ONNXTensorElementDataType dataType;
    checkStatus(pOrt_->GetTensorElementType(type_shape_ptr, &dataType));
    auto tensorType = ONNXTensorElementDataTypeToTensorProtoDataType(dataType);

    Tensor &output = (*output_tensors)[i];
    output.elem_type() = tensorType;

    // set tensor dims
    size_t num_dims;
    checkStatus(pOrt_->GetDimensionsCount(type_shape_ptr, &num_dims));
    std::vector<int64_t> output_dims(num_dims);
    pOrt_->GetDimensions(type_shape_ptr, output_dims.data(), num_dims);

    output.sizes().resize(num_dims);
    copy(output_dims.begin(), output_dims.end(), output.sizes().begin());

    // set tensor data
    size_t data_len = std::accumulate(output_dims.begin(), output_dims.end(),
                                      (int64_t)1, std::multiplies<int64_t>{});
    void *data_ptr;
    checkStatus(pOrt_->GetTensorMutableData(outputs[i], &(data_ptr)));
    SetTensorData(&output, data_ptr, data_len);
  }
  return true;
}

} // namespace inference
} // namespace ONNX_NAMESPACE
