// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the
// software in any medium, provided that you keep intact this entire notice. You may improve, modify and create
// derivative works of the software or any portion of the software, and you may copy and distribute such modifications
// or works. Modified works should carry a notice stating that you changed the software and should note the date and
// nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the
// source of the software. NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND,
// EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR
// WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE
// CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS
// THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE. You
// are solely responsible for determining the appropriateness of using and distributing the software and you assume
// all risks associated with its use, including but not limited to the risks and costs of program errors, compliance
// with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of 
// operation. This software is not intended to be used in any situation where a failure could cause risk of injury or
// damage to property. The software developed by NIST employees is not subject to copyright protection within the
// United States.


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

  static ColorScheme colorScheme; ///<< The color scheme to use for graph dot file
  static StructureOptions structureOptions; ///<< The structure options to use for graph dot file
  static DebugOptions debugOptions; ///<< The debug options to use for graph dot file

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
          signalString + "-graph-output.dot", colorScheme, structureOptions, debugOptions);
    }
  }

  /// @brief Create a dot file at exit if the instance still exist
  static void atExit() {
    if (graphInstance_)
      graphInstance_->createDotFile("Exit-graph-output.dot");
  }

  /// @brief Sets the color scheme for dot file generation
  /// @param scheme the color scheme
  static void setColorScheme(ColorScheme scheme) {
    colorScheme = scheme;
  }

  /// @brief Sets the structure options for dot file generation
  /// @param options the structure options
  static void setStructureOptions(StructureOptions options) {
    structureOptions = options;
  }

  /// @brief Sets the debug options for dot file generation
  /// @param options the debug options
  static void setDebugOptions(DebugOptions options) {
    debugOptions = options;
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


template<class GraphOutput, class ... GraphInputs>
ColorScheme GraphSignalHandler<GraphOutput, GraphInputs...>
    ::colorScheme = ColorScheme::EXECUTION; ///< Sets the default color scheme

template<class GraphOutput, class ... GraphInputs>
StructureOptions GraphSignalHandler<GraphOutput, GraphInputs...>
    ::structureOptions = StructureOptions::ALL; ///< Sets the default structure options

template<class GraphOutput, class ... GraphInputs>
DebugOptions GraphSignalHandler<GraphOutput, GraphInputs...>
    ::debugOptions = DebugOptions::ALL; ///< Sets the default debug options
}
#endif //HEDGEHOG_GRAPH_SIGNAL_HANDLER_H
