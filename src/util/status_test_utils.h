#include "macro_utils.h"

#define ASSERT_OK(s)         \
  auto TMPVAR(status) = (s); \
  ASSERT_TRUE(TMPVAR(status).ok()) << TMPVAR(status).status();

#define ASSERT_OK_AND_ASSIGN(decl, s)                           \
  auto TMPVAR(status) = s;                                      \
  ASSERT_TRUE(TMPVAR(status).ok()) << TMPVAR(status).status(); \
  decl = ::std::move(TMPVAR(status).value());
