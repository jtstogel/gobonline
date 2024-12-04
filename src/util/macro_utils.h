#define IMPL_CAT(a, b) IMPL_CAT_INNER(a, b)
#define IMPL_CAT_INNER(a, b) a##b

/**
 * TMPVAR(name) returns a variable name local to the macro invocation,
 * as long as `name` is unique within the macro.
 */
#define TMPVAR(name) IMPL_CAT(name##_tmp_, __LINE__)
