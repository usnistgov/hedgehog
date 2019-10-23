//
// Created by anb22 on 5/7/19.
//

#ifndef HEDGEHOG_GRAPH_SIGNAL_HANDLER_H
#define HEDGEHOG_GRAPH_SIGNAL_HANDLER_H
#pragma  once

#include <csignal>
#include <cstring>

#include "../../hedgehog.h"
/// @brief Hedgehog main namespace
namespace hh {
/// @brief Implements a signal handler to catch events such as termination and killing.
/// @details Once a signal is caught, all task graphs that are registered with the signal handler will be written as a
/// dot file. The dot file is output in the working directory with the name of the signal as a prefix and
/// '<#>-graph-output.dot' as the suffix.
/// @tparam GraphOutput Graph Output type
/// @tparam GraphInputs Graph Inputs type
template<class GraphOutput, class ... GraphInputs>
class GraphSignalHandler {
 private:
  static Graph<GraphOutput, GraphInputs...>
      *graphInstance_; ///<< The outer graph instance
  static bool
      signalHandled_; ///< Flag to indicate if a signal has been fired or not

 public:
  /// @brief Function that handles signals.
  /// @details Use TaskGraphSignalHandler::registerSignal to signal to this function
  /// @attention This function is used by the signal handler that is registered from std::signal, and should not be
  /// called directly by the user.
  /// @param signum the signal number that was triggered
  static void handleSignal(int signum = SIGTERM) {
#ifdef _WIN32
    std::string signalString(std::to_string(signum));
#else
    std::string signalString(strsignal(signum));
#endif

    if (!signalHandled_) {
      signalHandled_ = true;
      std::cout << "signal caught: " << signum << ": (" << signalString << ")" << std::endl;
      graphInstance_->createDotFile(
          signalString + "-graph-output.dot", ColorScheme::EXECUTION, StructureOptions::ALL, DebugOptions::DEBUG);
    }
  }

  /// @brief Create a dot file at exit if the instance still exist
  static void atExit() {
    if (graphInstance_)
      graphInstance_->createDotFile("Exit-graph-output.dot");
  }

  /// @brief Registers a task graph to be displayed when a signal is fired.
  /// @param graph the task graph to be displayed.
  static void registerGraph(Graph<GraphOutput, GraphInputs...> *graph) {
    graphInstance_ = graph;
  }

  /// @brief Registers a signal for handling. (default SIGTERM)
  /// @param signum Signal number id
  /// @param atExit Boolean to test if GraphSignalHandler::atExit is called
  static void registerSignal(int signum = SIGTERM, bool atExit = false) {
    std::signal(signum, GraphSignalHandler<GraphOutput, GraphInputs ...>::handleSignal);
    if (atExit) {
      std::atexit(GraphSignalHandler<GraphOutput, GraphInputs ...>::atExit);
    }
  }
};

template<class GraphOutput, class ... GraphInputs>
bool GraphSignalHandler<GraphOutput, GraphInputs...>
    ::signalHandled_ = false; ///< Set default value at false

template<class GraphOutput, class ... GraphInputs>
Graph<GraphOutput, GraphInputs...> *GraphSignalHandler<GraphOutput, GraphInputs...>
    ::graphInstance_ = nullptr; ///< Set default value at nullptr
}
#endif //HEDGEHOG_GRAPH_SIGNAL_HANDLER_H
