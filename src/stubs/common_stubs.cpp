// Stub implementations for common/ headers.
#include "common/assert.h"
#include "common/logging.h"

#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>

void AssertFailSoftImpl() {
    std::fprintf(stderr, "[NVN-STANDALONE] Assert failed (soft)\n");
#ifdef _DEBUG
    __debugbreak();
#endif
}

[[noreturn]] void AssertFatalImpl() {
    std::fprintf(stderr, "[NVN-STANDALONE] Fatal assert!\n");
    std::abort();
}

namespace Common::Log {

void FmtLogMessageImpl(Class /*log_class*/, Level log_level, const char* filename,
                       unsigned int line_num, const char* function,
                       fmt::string_view format, const fmt::format_args& args) {
    static constexpr const char* kLevelNames[] = {
        "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "CRIT"
    };
    const char* level_str = (unsigned)log_level < 6 ? kLevelNames[(unsigned)log_level] : "???";

    auto message = fmt::vformat(format, args);
    std::fprintf(stderr, "[%s] %s:%u (%s): %s\n", level_str, filename, line_num, function,
                 message.c_str());
}

} // namespace Common::Log
