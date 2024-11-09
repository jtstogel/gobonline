#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  std::cout << "Hello world!" << std::endl;

  if (random()) {
    return 1;
  } else {  // Shouldn't pass clang-tidy.
    return 0;
  }
}
