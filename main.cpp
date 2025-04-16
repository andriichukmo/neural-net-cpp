#include "except.h"
#include "tests.h"

int main() {
  try {
    Tests::test_basic();
    Tests::test_mnist();
  } catch (...) {
    Except::react();
  }
}
