//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_GRAPH_H
#define HEDGEHOG_GRAPH_H

#include <cassert>
#include <ostream>

#include "../behaviour/io/multi_receivers.h"
#include "../behaviour/io/sender.h"

#include "../core/node/core_graph.h"
#include "../core/scheduler/abstract_scheduler.h"

#include "../tools/helper.h"
#include "../tools/traits.h"
#include "../tools/printers/dot_printer.h"
#include "../tools/logger.h"

template<class GraphOutput, class ...GraphInputs>
class Graph :
    public MultiReceivers<GraphInputs...>,
    public Sender<GraphOutput>,
    public virtual Node {
  static_assert(HedgehogTraits::isUnique<GraphInputs...>, "A Graph can't accept multiple inputs with the same type.");
 private:
  std::shared_ptr<CoreGraph<GraphOutput, GraphInputs...>> graphCore_ = nullptr;
  std::set<std::shared_ptr<Node>> insideNodes_ = {};

 public:
  Graph() {
    graphCore_ = std::make_shared<CoreGraph<GraphOutput, GraphInputs...>>(this, NodeType::Graph, "Graph");
  }

  explicit Graph(std::string_view const &name) {
    graphCore_ = std::make_shared<CoreGraph<GraphOutput, GraphInputs...>>(this, NodeType::Graph, name);
  }

  ~Graph() override = default;

  std::shared_ptr<CoreNode> core() final { return this->graphCore_; }
  std::string_view const &name() { return this->core()->name(); }

  template<
      class UserDefinedMultiReceiver,
      class InputsMR = typename UserDefinedMultiReceiver::inputs_t,
      class InputsG = typename MultiReceivers<GraphInputs...>::inputs_t,
      class isMultiReceiver = typename std::enable_if<
          std::is_base_of_v<typename Helper::HelperMultiReceiversType<InputsMR>::type, UserDefinedMultiReceiver>
      >::type,
      class isInputCompatible = std::enable_if<HedgehogTraits::is_included_v<InputsMR, InputsG>>>
  void input(std::shared_ptr<UserDefinedMultiReceiver> input) {
    assert(input != nullptr);
    this->insideNodes_.insert(input);
    auto test = std::dynamic_pointer_cast<typename Helper::HelperMultiReceiversType<InputsMR>::type>(input);
    this->graphCore_->input(test);
  }

  template<
      class UserDefinedSender,
      class IsSender = typename std::enable_if<
          std::is_base_of_v<
              Sender<GraphOutput>, UserDefinedSender
          >
      >::type
  >
  void output(std::shared_ptr<UserDefinedSender> output) {
    assert(output != nullptr);
    this->insideNodes_.insert(output);
    this->graphCore_->output(std::static_pointer_cast<Sender<GraphOutput>>(output));
  }

  template<
      class UserDefinedSender, class UserDefinedMultiReceiver,
      class Output = typename UserDefinedSender::output_t,
      class Inputs = typename UserDefinedMultiReceiver::inputs_t,
      class IsSender = typename std::enable_if<std::is_base_of_v<Sender<Output>, UserDefinedSender>>::type,
      class IsMultiReceivers = typename std::enable_if<
          std::is_base_of_v<
              typename Helper::HelperMultiReceiversType<Inputs>::type, UserDefinedMultiReceiver
          >
      >::type
  >
  void addEdge(std::shared_ptr<UserDefinedSender> from, std::shared_ptr<UserDefinedMultiReceiver> to) {
    static_assert(HedgehogTraits::contains_v<Output, Inputs>,
                  "The given io cannot be linked to this io: No common types.");

    this->insideNodes_.insert(from);
    this->insideNodes_.insert(to);
    this->graphCore_->addEdge(std::static_pointer_cast<Sender<Output>>(from),
                              std::static_pointer_cast<typename Helper::HelperMultiReceiversType<Inputs>::type>(to));
  }

  void executeGraph() {
    this->graphCore_->executeGraph();
  }

  template<
      class Input,
      class = std::enable_if_t<HedgehogTraits::Contains<Input, GraphInputs...>::value>
  >
  void pushData(std::shared_ptr<Input> data) { this->graphCore_->broadcastAndNotifyToAllInputs(data); }

  void finishPushingData() { this->graphCore_->finishPushingData(); }

  std::shared_ptr<GraphOutput> getBlockingResult() {
    return this->graphCore_->getBlockingResult();
  }

  void waitForTermination() {
    this->graphCore_->waitForTermination();
  }

  void createDotFile(std::filesystem::path const &dotFilePath,
                     ColorScheme colorScheme = ColorScheme::NONE,
                     StructureOptions structureOptions = StructureOptions::NONE,
                     DebugOptions debugOption = DebugOptions::NONE) {
    auto core = this->core().get();
    DotPrinter printer(std::filesystem::absolute(dotFilePath), colorScheme, structureOptions, debugOption, core);
    core->visit(&printer);
  }
};

#endif //HEDGEHOG_GRAPH_H
