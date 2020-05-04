//=================================================================
// HB Profiler
// 05/03/2020 Bandhav Veluri
//=================================================================

#include <string>
#include <sstream>
#include <iomanip>
#include <c10/hammerblade/HammerBladeException.h>
#include <bsg_manycore_printing.h>
#include "HammerBladeProfiler.h"

namespace { // anonymous

// `bsg_time` to cycles conversion factor
const uint64_t cycle_time = 10;

uint64_t elapsed_cycles() {
  return bsg_time() / cycle_time;
}

template<typename T>
struct SummaryCol {
  std::string header;
  std::vector<T> contents;

  int width() {
    return header.length();
  }
};

template<>
int SummaryCol<std::string>::width() {
  int w = header.length();

  for(int i = 0; i < contents.size(); ++i) {
    if(contents[i].length() > w) {
      w = contents[i].length();
    }
  }

  return w;
}

} // namespace anonymous

namespace c10 {
namespace hammerblade {

void HBProfiler::kernel_start(const char* kernel) {
  if(profile_log.find(kernel) == profile_log.end()) {
    profile_log[kernel] = {0};
  }

  profile_log[kernel][START_TIME] = elapsed_cycles();
}

void HBProfiler::kernel_end(const char* kernel) {
  TORCH_INTERNAL_ASSERT(
    profile_log.find(kernel) != profile_log.end(),
    "Can't find the kernel in profiler_log. ",
    "Please call kernel_start first");

  profile_log[kernel][NUM_CALLS] += 1;
  profile_log[kernel][END_TIME] = elapsed_cycles();

  uint64_t kernel_exec_time = profile_log[kernel][1] -
                                profile_log[kernel][0];
  profile_log[kernel][CUMMULATIVE] += kernel_exec_time;
  execution_time += kernel_exec_time;
}

std::string HBProfiler::summary() {
  SummaryCol<std::string> names;
  SummaryCol<float> percents;
  SummaryCol<uint64_t> cycles;
  SummaryCol<uint64_t> num_calls;

  names.header = "Kernel Name";
  percents.header = "HB Total %";
  cycles.header = "HB Total Cycles";
  num_calls.header = "Number of Calls";

  for(auto k = profile_log.begin(); k != profile_log.end(); ++k) {
    names.contents.push_back(k->first);
    percents.contents.push_back(
        100.0 * ((float) k->second[CUMMULATIVE] / (float) execution_time));
    cycles.contents.push_back(k->second[CUMMULATIVE]);
    num_calls.contents.push_back(k->second[NUM_CALLS]);
  }

  std::stringstream summary;

  // Headers
  summary << std::setw(names.width() + 1) << names.header;
  summary << std::setw(percents.width() + 1) << percents.header;
  summary << std::setw(cycles.width() + 1) << cycles.header;
  summary << std::setw(num_calls.width() + 1) << num_calls.header;
  summary << std::endl;

  // Underlines
  summary << std::setw(names.width() + 1) << std::string("=", names.width());
  summary << std::setw(percents.width() + 1) << std::string("=", percents.width());
  summary << std::setw(cycles.width() + 1) << std::string("=", cycles.width());
  summary << std::setw(num_calls.width() + 1) << std::string("=", num_calls.width());
  summary << std::endl;

  std::string summary_string;
  summary >> summary_string;

  return summary_string;
}

}} // namespace c10::hammerblade
