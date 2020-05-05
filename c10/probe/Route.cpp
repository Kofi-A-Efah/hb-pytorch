#include <c10/probe/Route.h>
#include <c10/util/Exception.h>

#include <iostream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace c10 {
namespace probe {

ExecutionRoute g_execution_route;

// ========== ExecutionRoute Member ==========

void ExecutionRoute::reset() {
  route.clear();
  beacons.clear();
}

void ExecutionRoute::add_waypoint(const std::string& kernel, bool redispatch) {
  route.push_back(std::make_tuple(kernel, redispatch));
  beacons[kernel] = true;
}

void ExecutionRoute::print() {
  json chart_json = json::array();
  for (const auto& wp : route) {
    json kernel_json;
    kernel_json["signature"] = std::get<0>(wp);
    kernel_json["offload"] = std::get<1>(wp);
    chart_json.push_back(kernel_json);
  }
  std::cerr << chart_json.dump(4) << std::endl;
}

bool ExecutionRoute::should_redispatch(const std::string& kernel) {
  if (beacons.find(kernel) != beacons.end()) {
    std::cerr << "at top level kernel " << kernel << std::endl;
    TORCH_INTERNAL_ASSERT(odometer < route.size(), "ERROR: Route is shorter than execution chart");
    auto route_kernel = std::get<0>(route[odometer]);
    auto redispatch = std::get<1>(route[odometer]);
    TORCH_INTERNAL_ASSERT(route_kernel.compare(kernel) == 0,
        "ERROR: Route and execution chart disagree. Expect ", route_kernel, " but found ", kernel);
    odometer++;
    std::cerr << "should I redispatch? 1/0" << std::endl;
    if (redispatch) {
      std::cerr << "redispatching..." << std::endl;
      return true;
    }
  }
  return false;
}

// ========== ExecutionRoute C10_API ==========

void route_add_waypoint(const std::string& kernel, bool redispatch) {
  g_execution_route.add_waypoint(kernel, redispatch);
  // hack
  g_execution_route.print();
}

bool should_redispatch(const std::string& kernel) {
  return g_execution_route.should_redispatch(kernel);
}

}} // namespace c10::probe
