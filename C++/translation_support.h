#pragma once

#include <stdexcept>
#include <string>

namespace translation {

template <typename T>
T unsupported(const std::string& what) {
  throw std::runtime_error(what);
}

inline void unsupported_void(const std::string& what) {
  throw std::runtime_error(what);
}

} // namespace translation
