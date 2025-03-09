#include "except.h"
#include "tests.h"

int main() {
  try {
    Tests::run_all_tests();
  } catch (...) {
    Except::react();
  }
}
