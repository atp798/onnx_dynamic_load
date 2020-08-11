/**
 * Copyright (c) 2019, ATP
 * All rights reserved.
 * MIT License
 */

#pragma once

namespace Utilis {

class NonCopyable {
public:
  NonCopyable(NonCopyable const &) = delete;
  NonCopyable &operator=(NonCopyable const &) = delete;

protected:
  NonCopyable() = default;
  ~NonCopyable() = default;
};

} // namespace Utilis
