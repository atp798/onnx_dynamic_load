/* *
 * Copyright (c) 2020, ATP
 * All rights reserved.
 * MIT License
 */

#include "library_loader.h"

#include <cstdlib>
#include <fstream>

namespace ONNX_NAMESPACE {
namespace inference {

void dlHandleDeleter(void *handler) {
  if (nullptr == handler)
    return;
  dlclose(handler);
}

const LibraryLoader &LibraryLoader::GetInstance() {
  static LibraryLoader loader_;
  return loader_;
}

LibraryLoader::LibraryLoader() {
  std::string path_env(getenv(kLdLibraryPath.c_str()));

  std::string delimiters = ":";
  std::string::size_type start_pos = path_env.find_first_not_of(delimiters, 0);
  std::string::size_type end_pos =
      path_env.find_first_of(delimiters, start_pos);
  while (std::string::npos != start_pos) {
    if (std::string::npos != end_pos) {
      libPaths_.emplace_back(path_env.substr(start_pos, end_pos - start_pos));
    } else {
      libPaths_.emplace_back(path_env.substr(start_pos));
      break;
    }
    start_pos = path_env.find_first_not_of(delimiters, end_pos);
    end_pos = path_env.find_first_of(delimiters, start_pos);
  }
}

std::string LibraryLoader::findLibrary(const std::string &lib_name) const {
  std::string lib_name_tmp = lib_name;
  if (lib_name_tmp.find('/') != 0) {
    lib_name_tmp = "/" + lib_name_tmp;
  }
  std::string lib_path;
  for (auto &path : libPaths_) {
    std::string tmp = path + lib_name_tmp;
    std::fstream f;
    f.open(tmp);
    if (f.is_open()) {
      lib_path.swap(tmp);
      f.close();
      break;
    }
  }
  return lib_path;
}

LibHandlePtr LibraryLoader::LoadLibrary(const std::string &lib_name) const
    throw(std::runtime_error) {
  std::string lib_path = findLibrary(lib_name);
  if (lib_path.length() == 0) {
    std::ostringstream err_msg;
    err_msg << "Library: '" << lib_name << "' not found.";
    throw std::runtime_error(err_msg.str());
  }

  // dlopen handle
  void *dl_handle = dlopen(lib_path.c_str(), RTLD_LAZY);
  char *error = dlerror();
  if (nullptr != error) {
    std::ostringstream err_msg;
    err_msg << "Library: '" << lib_name << "' load failed.";
    throw std::runtime_error(err_msg.str());
  }
  LibHandlePtr sp_handle(dl_handle, dlHandleDeleter);
  return std::move(sp_handle);
}
} // namespace inference
} // namespace ONNX_NAMESPACE
