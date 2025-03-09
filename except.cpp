#pragma once

#include "except.h"
#include <exception>
#include <iostream>

namespace Except {

void react() {
  try {
    throw;
  } catch (const std::exception &e) {
    std::cerr << "Caught exceptiom: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Caught unknown exception" << std::endl;
  }
}

} // namespace Except
