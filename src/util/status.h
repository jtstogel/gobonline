#include "macro_utils.h"

/**
 * Returns the expressions status if it is an error.
 */
#define RETURN_IF_ERROR(status_expression)   \
  auto TMPVAR(statusor) = status_expression; \
  if (!TMPVAR(statusor).ok()) return TMPVAR(tmp).status()

/**
 * Assigns the return value of `status_expression` to `decl`,
 * if the status is OK. Else, returns the error status.
 */
#define ASSIGN_OR_RETURN(decl, status_expression)               \
  auto TMPVAR(statusor) = status_expression;                    \
  /** There's probably a way to avoid default initializing. */  \
  if (!TMPVAR(statusor).ok()) return TMPVAR(statusor).status(); \
  decl = ::std::move(TMPVAR(statusor).value());
