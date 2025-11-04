#include "logger.h"
#include <iostream>
#include <vector>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace Logger {

static std::shared_ptr<spdlog::logger> global_logger;

void init(const std::string& log_file) {
    try {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, true);

        std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};

        global_logger = std::make_shared<spdlog::logger>("app_logger", sinks.begin(), sinks.end());
        spdlog::register_logger(global_logger);

        // Set format: [2025-11-03 14:10:22] [info] Message
        global_logger->set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");

        // Set log level (trace/debug/info/warn/error/critical/off)
        global_logger->set_level(spdlog::level::debug);
        global_logger->flush_on(spdlog::level::info);

        spdlog::set_default_logger(global_logger);
    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Log initialization failed: " << ex.what() << std::endl;
    }
}

std::shared_ptr<spdlog::logger> get() {
    if (!global_logger)
        throw std::runtime_error("Logger not initialized! Call Logger::init() first.");
    return global_logger;
}

} // namespace Logger
