// Stub: common/logging.h — Logging macros that forward to fmt + stderr.
// In the standalone project, logging goes to stderr instead of Eglantine's log subsystem.
#pragma once

#include <chrono>
#include <algorithm>
#include <string_view>
#include <type_traits>
#include <fmt/format.h>
#include "common/swap.h"

// Enum formatter (matches Eglantine's)
#if FMT_VERSION >= 80100
template <typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_enum_v<T>, char>>
    : formatter<std::underlying_type_t<T>> {
    template <typename FormatContext>
    auto format(const T& value, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::formatter<std::underlying_type_t<T>>::format(
            static_cast<std::underlying_type_t<T>>(value), ctx);
    }
};
#endif

template <typename T, typename U>
struct fmt::formatter<SwapStructT<T, U>> {
    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }
    template <typename FormatContext>
    auto format(const SwapStructT<T, U>& reg, FormatContext& ctx) const {
        return fmt::format_to(ctx.out(), "{}", T(reg));
    }
};

#ifdef _DEBUG
#define LOG_TRACE(log_class, ...) \
    Common::Log::FmtLogMessage(Common::Log::Class::log_class, Common::Log::Level::Trace, \
       __FILE__, __LINE__, __func__, __VA_ARGS__)
#else
#define LOG_TRACE(log_class, fmt, ...) (void(0))
#endif

#define LOG_DEBUG(log_class, ...) \
    ::Common::Log::FmtLogMessage(::Common::Log::Class::log_class, ::Common::Log::Level::Debug, \
       __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_INFO(log_class, ...) \
    ::Common::Log::FmtLogMessage(::Common::Log::Class::log_class, ::Common::Log::Level::Info, \
       __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_WARNING(log_class, ...) \
    ::Common::Log::FmtLogMessage(::Common::Log::Class::log_class, ::Common::Log::Level::Warning, \
       __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_ERROR(log_class, ...) \
    ::Common::Log::FmtLogMessage(::Common::Log::Class::log_class, ::Common::Log::Level::Error, \
       __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_CRITICAL(log_class, ...) \
    ::Common::Log::FmtLogMessage(::Common::Log::Class::log_class, ::Common::Log::Level::Critical, \
       __FILE__, __LINE__, __func__, __VA_ARGS__)

namespace Common::Log {

enum class Level : u8 {
    Trace,
    Debug,
    Info,
    Warning,
    Error,
    Critical,
    Count
};

enum class Class : u8 {
#define SUB(A, B) A##_##B,
#define CLS(A) A,
    CLS(Log)
    CLS(Common)
    SUB(Common, Filesystem)
    SUB(Common, Memory)
    CLS(Core)
    SUB(Core, ARM)
    SUB(Core, Timing)
    CLS(Config)
    CLS(Debug)
    SUB(Debug, Emulated)
    SUB(Debug, GPU)
    SUB(Debug, Breakpoint)
    SUB(Debug, GDBStub)
    SUB(Debug, MCP)
    CLS(Kernel)
    SUB(Kernel, SVC)
    CLS(Service)
    SUB(Service, ACC)
    SUB(Service, Audio)
    SUB(Service, AM)
    SUB(Service, AOC)
    SUB(Service, APM)
    SUB(Service, ARP)
    SUB(Service, BCAT)
    SUB(Service, BPC)
    SUB(Service, BGTC)
    SUB(Service, BTDRV)
    SUB(Service, BTM)
    SUB(Service, Capture)
    SUB(Service, ERPT)
    SUB(Service, ETicket)
    SUB(Service, EUPLD)
    SUB(Service, Fatal)
    SUB(Service, FGM)
    SUB(Service, Friend)
    SUB(Service, FS)
    SUB(Service, GRC)
    SUB(Service, HID)
    SUB(Service, IRS)
    SUB(Service, JIT)
    SUB(Service, LBL)
    SUB(Service, LDN)
    SUB(Service, LDR)
    SUB(Service, LM)
    SUB(Service, Migration)
    SUB(Service, Mii)
    SUB(Service, MM)
    SUB(Service, MNPP)
    SUB(Service, NCM)
    SUB(Service, NFC)
    SUB(Service, NFP)
    SUB(Service, NGC)
    SUB(Service, NIFM)
    SUB(Service, NIM)
    SUB(Service, NOTIF)
    SUB(Service, NPNS)
    SUB(Service, NS)
    SUB(Service, NVDRV)
    SUB(Service, Nvnflinger)
    SUB(Service, OLSC)
    SUB(Service, PCIE)
    SUB(Service, PCTL)
    SUB(Service, PCV)
    SUB(Service, PM)
    SUB(Service, PREPO)
    SUB(Service, PSC)
    SUB(Service, PTM)
    SUB(Service, SET)
    SUB(Service, SM)
    SUB(Service, SPL)
    SUB(Service, SSL)
    SUB(Service, TCAP)
    SUB(Service, Time)
    SUB(Service, USB)
    SUB(Service, VI)
    SUB(Service, WLAN)
    CLS(HW)
    SUB(HW, Memory)
    SUB(HW, LCD)
    SUB(HW, GPU)
    SUB(HW, AES)
    CLS(IPC)
    CLS(Frontend)
    CLS(Render)
    SUB(Render, Software)
    SUB(Render, OpenGL)
    SUB(Render, Vulkan)
    CLS(Shader)
    SUB(Shader, SPIRV)
    SUB(Shader, GLASM)
    SUB(Shader, GLSL)
    CLS(Audio)
    SUB(Audio, DSP)
    SUB(Audio, Sink)
    CLS(Input)
    CLS(Network)
    CLS(Loader)
    CLS(CheatEngine)
    CLS(Crypto)
    CLS(WebService)
#undef SUB
#undef CLS
    Count,
};

void FmtLogMessageImpl(Class log_class, Level log_level, const char* filename,
                       unsigned int line_num, const char* function,
                       fmt::string_view format, const fmt::format_args& args);

[[nodiscard]] inline bool IsLevelEnabled(Class /*log_class*/, Level /*log_level*/) {
    return true;
}

template <typename... Args>
void FmtLogMessage(Class log_class, Level log_level, const char* filename,
                   unsigned int line_num, const char* function,
                   fmt::format_string<Args...> format, const Args&... args) {
    FmtLogMessageImpl(log_class, log_level, filename, line_num, function,
                      format.get(), fmt::make_format_args(args...));
}

inline void Initialize() {}
inline void Start() {}
inline void Stop() {}

} // namespace Common::Log
