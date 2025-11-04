#pragma once
#include <memory>
#include <spdlog/spdlog.h>

namespace Logger {

// Initialize logger once at startup
void init(const std::string& log_file = "logs/app.log");

// Get a global logger instance
std::shared_ptr<spdlog::logger> get();

} // namespace Logger
