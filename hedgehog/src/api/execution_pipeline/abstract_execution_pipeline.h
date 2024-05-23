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

#ifndef HEDGEHOG_ABSTRACT_EXECUTION_PIPELINE_H
#define HEDGEHOG_ABSTRACT_EXECUTION_PIPELINE_H

#include "../graph/graph.h"
#include "../../behavior/copyable.h"
#include "../../core/nodes/core_execution_pipeline.h"
#include "../../behavior/switch/multi_switch_rules.h"

/// @brief Hedgehog main namespace
namespace hh {

/// Execution pipeline abstraction
/// @brief Duplicate a graph with the same input and output types and associate each of the duplicates to a specified
/// device id (GPU). If none is provided, the devices are generated in sequence (for 3 duplicates, 4 graphs total, the
/// device ids are 0,1,2,3).
/// When implementing, the switch rules need to be provided. They are used to redirect an input data sent to the
/// execution pipeline to a specific graph. Each of the graphs inside an execution pipeline has an id (generated in
/// sequence) used to discriminate the graphs in the switch rules.
/// If the Execution pipeline is duplicated (because it is part of a graph which is also in another execution pipeline),
/// the copy method needs to be implemented.
/// @code
/// // Implementation of an execution pipeline that accepts int, float and double data and produces int, float and double data.
/// class IntFloatDoubleExecutionPipeline
///    : public hh::AbstractExecutionPipeline<3, int, float, double, int, float, double> {
/// public:
///  IntFloatDoubleExecutionPipeline(std::shared_ptr<hh::Graph<3, int, float, double, int, float, double>> const &graph,
///                                  size_t const &numberGraphs)
///      : hh::AbstractExecutionPipeline<3, int, float, double, int, float, double>(graph, numberGraphs) {}
///
///  ~IntFloatDoubleExecutionPipeline() override = default;
///
///  bool sendToGraph(std::shared_ptr<int> &data, size_t const &graphId) override {
///    // Return true of false if the int data needs to be sent to the graph of id graphId
///  }
///
///  bool sendToGraph(std::shared_ptr<float> &data, size_t const &graphId) override {
///    // Return true of false if the float data needs to be sent to the graph of id graphId
///  }
///
///  bool sendToGraph(std::shared_ptr<double> &data, size_t const &graphId) override {
///    // Return true of false if the double data needs to be sent to the graph of id graphId
///  }
///};
/// // Instantiate a graph and the execution pipeline
///  auto insideGraph = std::make_shared<hh::Graph<3, int, float, double, int, float, double>>();
///  auto innerInputInt = std::make_shared<IntFloatDoubleTask>();
///  auto innerTaskFloat = std::make_shared<IntFloatDoubleTask>();
///  auto innerOutput = std::make_shared<IntFloatDoubleTask>();
///  auto innerSM = std::make_shared<hh::StateManager<1, int, int>>(std::make_shared<IntState>());
///  auto innerGraph = std::make_shared<IntFloatDoubleGraph>();
///
/// // Create a graph
///  insideGraph->input<int>(innerInputInt);
///  insideGraph->input<float>(innerTaskFloat);
///  insideGraph->inputs(innerSM);
///  insideGraph->inputs(innerGraph);
///
///  insideGraph->edges(innerInputInt, innerOutput);
///  insideGraph->edges(innerSM, innerOutput);
///  insideGraph->edges(innerTaskFloat, innerOutput);
///  insideGraph->outputs(innerOutput);
///  insideGraph->outputs(innerGraph);
///
///  auto ep = std::make_shared<IntFloatDoubleExecutionPipeline>(insideGraph, 5);
/// @endcode
/// @attention The duplicated graph needs to be totally created before set to an execution pipeline, it won't be
/// modifiable thereafter.
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template<size_t Separator, class ...AllTypes>
class AbstractExecutionPipeline
    : public behavior::Node,
      public behavior::Copyable<AbstractExecutionPipeline<Separator, AllTypes...>>,
      public tool::BehaviorMultiReceiversTypeDeducer_t<tool::Inputs<Separator, AllTypes...>>,
      public tool::BehaviorMultiSwitchRulesTypeDeducer_t<tool::Inputs<Separator, AllTypes...>>,
      public tool::BehaviorMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>> {
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  /// @brief Declare core::CoreExecutionPipeline as friend
  friend hh::core::CoreExecutionPipeline<Separator, AllTypes...>;
#endif //DOXYGEN_SHOULD_SKIP_THIS
 private:
  std::shared_ptr<Graph<Separator, AllTypes...>>
      graph_ = nullptr; ///< Original Graph that will be duplicated

  std::shared_ptr<core::CoreExecutionPipeline<Separator, AllTypes...>> const
      coreExecutionPipeline_ = nullptr; ///< Execution Pipeline core
 public:
  /// Create an execution pipeline that duplicates a @p graph, @p numberGraphs - 1 times. The graph id and the device id
  /// associated to each graph are generated in sequence. The given name is "Execution pipeline" by default.
  /// @param graph Graph to duplicate
  /// @param numberGraphs Number of graph in total in the execution pipeline
  /// @param name Name of the execution pipeline
  AbstractExecutionPipeline(
      std::shared_ptr<Graph<Separator, AllTypes...>> const graph,
      size_t const &numberGraphs,
      std::string const name = "Execution pipeline")
      : behavior::Node(
      static_cast<std::shared_ptr<core::abstraction::NodeAbstraction>>(
          static_cast<std::shared_ptr<core::abstraction::NodeAbstraction> const>(
              std::make_shared<core::CoreExecutionPipeline<Separator, AllTypes...>>(
                  this, graph->coreGraph_, numberGraphs, name)))
  ),
        behavior::Copyable<AbstractExecutionPipeline<Separator, AllTypes...>>(1),
        graph_(graph),
        coreExecutionPipeline_(std::dynamic_pointer_cast<core::CoreExecutionPipeline<Separator,
                                                                                     AllTypes...>>(this->core())) {}

  /// Create an execution pipeline from a graph and the given device ids. If there are n device ids given, the graph
  /// will be duplicated n-1 times (for a total of n graphs), and each device ids will be associated to each graph.
  /// The given name is "Execution pipeline" by default.
  /// @param graph Graph to duplicate
  /// @param deviceIds Vector of device ids associated to the graphs in the execution pipeline
  /// @param name Name of the execution pipeline
  AbstractExecutionPipeline(
      std::shared_ptr<Graph<Separator, AllTypes...>> const graph,
      std::vector<int> const &deviceIds,
      std::string const name = "Execution pipeline")
      : behavior::Node(
      std::make_shared<core::CoreExecutionPipeline<Separator, AllTypes...>>(this, graph->coreGraph_, deviceIds, name)
  ),
        behavior::Copyable<AbstractExecutionPipeline<Separator, AllTypes...>>(1),
        graph_(graph),
        coreExecutionPipeline_(std::dynamic_pointer_cast<core::CoreExecutionPipeline<Separator,
                                                                                     AllTypes...>>(this->core())) {}

  /// Default destructor
  ~AbstractExecutionPipeline() override = default;

 private:
  /// Accessor to the base graph
  /// @return The base Graph
  std::shared_ptr<Graph<Separator, AllTypes...>> const &graph() const { return graph_; }

  /// @brief graph setter
  /// @param graph Graph to set
  void graph(std::shared_ptr<Graph<Separator, AllTypes...>> graph) { graph_ = graph; }
};

}

#endif //HEDGEHOG_ABSTRACT_EXECUTION_PIPELINE_H
