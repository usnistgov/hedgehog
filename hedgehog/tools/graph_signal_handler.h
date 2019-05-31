//
// Created by anb22 on 5/7/19.
//

#ifndef HEDGEHOG_GRAPH_SIGNAL_HANDLER_H
#define HEDGEHOG_GRAPH_SIGNAL_HANDLER_H

#include "../hedgehog.h"
#include <csignal>
#include <cstring>
template<class GraphOutput, class ... GraphInputs>
class GraphSignalHandler {
 private:
  static Graph<GraphOutput, GraphInputs...> *graphInstance_; //!<< The outer graph instance
  static bool signalHandled_; // !< Flag to indicate if a signal has been fired or not

 public:
  /**
    * Function that handles signals.
    * Use TaskGraphSignalHandler::registerSignal to signal to this function.
    * @param signum the signal number that was triggered
    */
  static void handleSignal(int signum = SIGTERM) {
#ifdef _WIN32
    std::string signalString(std::to_string(signum));
#else
    std::string signalString(strsignal(signum));

#endif
    if (!signalHandled_) {

      signalHandled_ = true;

      std::cout << "signal caught: " << signum << ": (" << signalString << ")" << std::endl;
      graphInstance_->createDotFile(signalString + "-graph-output.dot");
//      exit(signum);
    }
  }

  static void atExit() {
    if (graphInstance_)
      graphInstance_->createDotFile("Exit-graph-output.dot");
  }

  /**
   * Registers a task graph to be displayed when a signal is fired.
   * @param graph the task graph to be displayed.
   */
  static void registerGraph(Graph<GraphOutput, GraphInputs...> *graph) {
    graphInstance_ = graph;
  }

  /**
   * Registers a signal for handling. (default SIGTERM)
   * @param signum
   */
  static void registerSignal(int signum = SIGTERM, bool atExit = false) {
    std::signal(signum, GraphSignalHandler<GraphOutput, GraphInputs ...>::handleSignal);
    if (atExit) {
      std::atexit(GraphSignalHandler<GraphOutput, GraphInputs ...>::atExit);
    }
  }
};

template<class GraphOutput, class ... GraphInputs>
bool GraphSignalHandler<GraphOutput, GraphInputs...>::signalHandled_ = false;

template<class GraphOutput, class ... GraphInputs>
Graph<GraphOutput, GraphInputs...> *GraphSignalHandler<GraphOutput, GraphInputs...>::graphInstance_ = nullptr;

#endif //HEDGEHOG_GRAPH_SIGNAL_HANDLER_H
