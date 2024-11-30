#include "macro_utils.h"

/**
 * Returns the expressions status if it is an error.
 */
#define RETURN_IF_ERROR(status_expression) \
  auto IMPL_TMPVAR = status_expression;    \
  if (!IMPL_TMPVAR.ok()) return IMPL_TMPVAR.status()

/**
 * Assigns the return value of `status_expression` to `decl`,
 * if the status is OK. Else, returns the error status.
 */
#define ASSIGN_OR_RETURN(decl, status_expression)              \
  auto IMPL_TMPVAR = status_expression;                        \
  /** There's probably a way to avoid default initializing. */ \
  if (!IMPL_TMPVAR.ok()) return IMPL_TMPVAR.status();          \
  decl = ::std::move(IMPL_TMPVAR.value());
