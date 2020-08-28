/* *
 * Copyright (c) 2020, ATP
 * All rights reserved.
 * MIT License
 */

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "library_loader.h"
#include "onnx/common/ir.h"
#include "onnx/onnx_pb.h"
#include "utilis/NonCopyable.h"

struct OrtStatus;
struct OrtApiBase;
struct OrtApi;

namespace ONNX_NAMESPACE {
namespace inference {

class OnnxApiWrapper;
using OnnxApiWrapperPtr = std::shared_ptr<OnnxApiWrapper>;

// This class is non-thread-safe.
class OnnxApiWrapper : Utilis::NonCopyable {
public:
  static OnnxApiWrapperPtr GetInstance();
  OnnxApiWrapper() = delete;
  OnnxApiWrapper(LibHandlePtr lib_handle_sptr, const OrtApi *ort_ptr) noexcept
      : spLibHandle_(lib_handle_sptr), pOrt_(ort_ptr) {}

  bool InferModel(const ModelProto mp,
                  const std::vector<std::string> &input_names,
                  const std::vector<Tensor> &input_tensors,
                  const std::vector<std::string> &output_names,
                  std::vector<Tensor> *output_tensors);

private:
  // Non-copyable
  OnnxApiWrapper(const OnnxApiWrapper &) = delete;
  OnnxApiWrapper &operator=(const OnnxApiWrapper &) = delete;

  void checkStatus(OrtStatus *status) const;

  static std::weak_ptr<OnnxApiWrapper> wpWrapper_;
  static std::mutex muxWrapper_;

  static const std::string kOnnxruntimeLibName;
  LibHandlePtr spLibHandle_;

  static const std::string kOrtGetApiBaseName;
  typedef const OrtApiBase *(OrtGetApiBasePtr)();

  // The lifetime of pOrt_ is as long as the onnxruntime library in memory.
  // That means same as spLibHandle_, and same as the class instance.
  const OrtApi *const pOrt_;
};

} // namespace inference
} // namespace ONNX_NAMESPACE
