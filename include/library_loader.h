/* *
 * Copyright (c) 2020, ATP
 * All rights reserved.
 * MIT License
 */

#pragma once

#include <dlfcn.h>

#include <cstdio>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "utilis/NonCopyable.h"

namespace ONNX_NAMESPACE {
namespace inference {

void handle_ptrDeleter(void *handler);

using LibHandlePtr = std::shared_ptr<void>;

// This is thread-safe.
class LibraryLoader : Utilis::NonCopyable {
public:
  static const LibraryLoader &GetInstance();
  LibraryLoader();
  LibHandlePtr LoadLibrary(const std::string &lib_name) const;

  template <class FuncT>
  FuncT *GetFuncPointer(LibHandlePtr handle_ptr,
                        const std::string &func_name) const;

private:
  std::string findLibrary(const std::string &lib_name) const;

  const std::string kLdLibraryPath = "LD_LIBRARY_PATH";
  std::vector<std::string> libPaths_;
};

template <class FuncT>
FuncT *LibraryLoader::GetFuncPointer(LibHandlePtr handle_ptr,
                                     const std::string &func_name) const {
  FuncT *ptr =
      reinterpret_cast<FuncT *>(dlsym(handle_ptr.get(), func_name.c_str()));
  char *error = dlerror();
  if (nullptr != error) {
    std::ostringstream err_msg("Warning: dlsym() find func failed: ");
    err_msg << error << ".\n";
    throw std::runtime_error(err_msg.str());
  }
  return ptr;
}

} // namespace inference
} // namespace ONNX_NAMESPACE
