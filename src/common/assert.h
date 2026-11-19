// Stub: common/assert.h — Assertion macros (standalone version).
#pragma once

#include "common/logging.h"

void AssertFailSoftImpl();
[[noreturn]] void AssertFatalImpl();

#ifdef _MSC_VER
#define YUZU_NO_INLINE __declspec(noinline)
#else
#define YUZU_NO_INLINE __attribute__((noinline))
#endif

#define ASSERT_MSG(_a_, ...)                                                                       \
    ([&]() YUZU_NO_INLINE {                                                                         \
        auto&& assert_condition = (_a_);                                                           \
        if (!(assert_condition)) [[unlikely]] {                                                   \
            LOG_CRITICAL(Debug, __FILE__ ": assert " __VA_ARGS__);                                \
            AssertFailSoftImpl();                                                                  \
        }                                                                                          \
    }())
#define ASSERT(_a_) ASSERT_MSG(_a_, "{}", #_a_)

#define UNREACHABLE_MSG(...)                                                                       \
    do {                                                                                           \
        LOG_CRITICAL(Debug, __FILE__ ": unreachable " __VA_ARGS__);                               \
        AssertFatalImpl();                                                                         \
    } while (0)
#define UNREACHABLE() UNREACHABLE_MSG("")

#ifdef _DEBUG
#define DEBUG_ASSERT(_a_) ASSERT(_a_)
#define DEBUG_ASSERT_MSG(_a_, ...) ASSERT_MSG(_a_, __VA_ARGS__)
#else
#define DEBUG_ASSERT(_a_)                                                                          \
    do {                                                                                           \
    } while (0)
#define DEBUG_ASSERT_MSG(_a_, _desc_, ...)                                                         \
    do {                                                                                           \
    } while (0)
#endif

#define UNIMPLEMENTED() ASSERT(false && "Unimplemented!")
#define UNIMPLEMENTED_MSG(...) ASSERT_MSG(false, __VA_ARGS__)

#define UNIMPLEMENTED_IF(cond) ASSERT((!(cond)) && "Unimplemented!")
#define UNIMPLEMENTED_IF_MSG(cond, ...) ASSERT_MSG(!(cond), __VA_ARGS__)

#define ASSERT_OR_EXECUTE_MSG(_a_, _b_, ...)                                                       \
    do {                                                                                           \
        ASSERT_MSG(_a_, __VA_ARGS__);                                                              \
        if (!(_a_)) { _b_ }                                                                        \
    } while (0)

#define ASSERT_OR_EXECUTE(_a_, _b_) ASSERT_OR_EXECUTE_MSG(_a_, _b_, "{}", #_a_)
