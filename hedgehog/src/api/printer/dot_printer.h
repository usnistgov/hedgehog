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

#ifndef HEDGEHOG_DOT_PRINTER_H
#define HEDGEHOG_DOT_PRINTER_H

#include <fstream>
#include <filesystem>
#include <iostream>
#include <cmath>
#include <utility>
#include <set>

#include "printer.h"
#include "options/color_scheme.h"
#include "options/color_picker.h"
#include "options/debug_options.h"
#include "options/structure_options.h"
#include "../../core/abstractions/base/node/graph_node_abstraction.h"
#include "../../core/abstractions/base/node/task_node_abstraction.h"
#include "../../core/abstractions/base/any_groupable_abstraction.h"
#include "options/input_option.h"
#include "../../core/abstractions/base/node/state_manager_node_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {

/// @brief Printer to produce a dot representation of the current state of the graph
/// @details https://www.graphviz.org/doc/info/lang.html
class DotPrinter : public Printer {
 private:
  /// @brief Representation of an edge for the Dot Printer
  class Edge {
   private:
    std::string id_{}; ///< Edge id

    std::string type_{}; ///< Edge type

    std::string extraLabel_{}; ///< Edge extra label (QS / MQS ...)

    std::set<std::string>
        arrivals_{}, ///< Arrival points (sender nodes) for the edge
    exits_{}; ///< Exit points (receiver nodes) for the edge

    bool declarationPrinted_ = false; ///< Flag, true if edge declaration printed

    core::abstraction::GraphNodeAbstraction *
        belongingGraph_ = nullptr; ///< Graph owning the edge

   public:
    /// @brief Edge constructor
    /// @param id Edge id
    /// @param type Edge type
    /// @param belongingGraph Graph owning the edge
    explicit Edge(std::string id, std::string type, core::abstraction::GraphNodeAbstraction *const belongingGraph)
        : id_(std::move(id)), type_(std::move(type)), belongingGraph_(belongingGraph) {}

    /// @brief Edge destructor
    virtual ~Edge() = default;

    /// @brief Edge id accessor
    /// @return Edge id
    [[nodiscard]] std::string const &id() const { return id_; }

    /// @brief Extra label accessor
    /// @return Extra label
    [[nodiscard]] std::string const &extraLabel() const { return extraLabel_; }

    /// @brief Belonging graph accessor
    /// @return Belonging graph
    [[nodiscard]] core::abstraction::GraphNodeAbstraction *belongingGraph() const { return belongingGraph_; }

    /// @brief Extra label setter
    /// @param extraLabel Extra label to set
    void addExtraLabel(std::string extraLabel) { extraLabel_ = std::move(extraLabel); }

    /// @brief Arrival point register
    /// @param arrival Arrival point to register
    void addArrival(std::string arrival) { arrivals_.insert(std::move(arrival)); }

    /// @brief Exit point register
    /// @param exit Exit point to register
    void addExit(std::string exit) { exits_.insert(std::move(exit)); }

    /// @brief Print declaration of an edge
    /// @param os Output stream used to print the edge declaration
    void printDeclaration(std::ostream &os) {
      if (!declarationPrinted_) {
        declarationPrinted_ = true;
        os << "\"" << id_ << "\"" << "[label=\"" << type_ << "\\n" << extraLabel_ << "\", shape=rect];\n";
      }
    }

    /// @brief Print all edges parts in the dot format
    /// @param os Output stream used to print all edges parts in the dot format
    void printEdges(std::ostream &os) const {
      for (auto &arrival : arrivals_) {
        os << "\"" << arrival << "\" -> \"" << id_ << "\"[penwidth=1, dir=none];\n";
      }
      for (auto &exit : exits_) {
        os << "\"" << id_ << "\" -> \"" << exit << "\"[penwidth=1];\n";
      }
    }

    /// @brief Equality operator
    /// @param rhs Edge to test against
    /// @return True if the two edges (this and rhs) are considered the same, else False
    bool operator==(Edge const &rhs) const { return id_ == rhs.id_ && type_ == rhs.type_; }
  };

  std::ofstream outputFile_ = {}; ///< Output file stream
  ColorScheme colorScheme_ = {}; ///< Color scheme chosen
  StructureOptions structureOptions_ = {}; ///< Structure options chosen
  InputOptions inputOptions_ = {}; ///< Input option chosen
  DebugOptions debugOptions_ = {}; ///< Debug option chosen
  std::unique_ptr<ColorPicker> colorPicker_ = nullptr; ///< Color picker used to generate the dot file

  std::chrono::nanoseconds
      graphTotalExecution_ = std::chrono::nanoseconds::max(), ///< Total graph execution
  minExecutionDurationInAllGraphs_ =
  std::chrono::nanoseconds::max(), ///< Minimum execution duration among all nodes in the graph
  maxExecutionDurationInAllGraphs_ =
  std::chrono::nanoseconds::min(), ///< Maximum execution duration among all nodes in the graph
  rangeExecutionDurationInAllGraphs_ =
  std::chrono::nanoseconds::min(),  ///< Execution duration range among all nodes in the graph
  minWaitDurationInAllGraphs_ =
  std::chrono::nanoseconds::max(),  ///< Minimum wait duration among all nodes in the graph
  maxWaitDurationInAllGraphs_ =
  std::chrono::nanoseconds::min(),  ///< Maximum wait duration among all nodes in the graph
  rangeWaitDurationInAllGraphs_ =
  std::chrono::nanoseconds::min();  ///< Execution wait range among all nodes in the graph

  std::unique_ptr<std::vector<Edge>> edges_ = nullptr; ///< All edges in the graph

 public:
  /// @brief DotPrinter constructor
  /// @param dotFilePath Path for the generated dot file
  /// @param colorScheme Color scheme options (Color depending on exec time or wait time)
  /// @param structureOptions Structure options
  /// @param debugOptions Debug options
  /// @param graph Graph to represent
  /// @param colorPicker Range of colors used to generate the dot file
  /// @param verbose Enable verbose mode: report when dot files are created or overwritten to standard out, default
  /// false.
  /// @throw std::runtime_error if the dot printer is not constructed with a valid ColorPicker
  DotPrinter(std::filesystem::path const &dotFilePath,
             ColorScheme colorScheme,
             StructureOptions structureOptions,
             InputOptions inputOptions,
             DebugOptions debugOptions,
             core::abstraction::GraphNodeAbstraction const *graph,
             std::unique_ptr<ColorPicker> colorPicker,
             bool verbose)
      : colorScheme_(colorScheme),
        structureOptions_(structureOptions),
        inputOptions_(inputOptions),
        debugOptions_(debugOptions),
        colorPicker_(std::move(colorPicker)),
        edges_(std::make_unique<std::vector<Edge>>()) {
    assert(graph != nullptr);

    testPath(dotFilePath, verbose);

    if (colorPicker_ == nullptr) {
      throw (
          std::runtime_error("A dot printer should be constructed with a valid ColorPicker (colorPicker != nullptr)")
      );
    }

    auto minMaxExecTime = graph->minMaxExecutionDuration();
    auto minMaxWaitTime = graph->minMaxWaitDuration();

    minExecutionDurationInAllGraphs_ = minMaxExecTime.first;
    maxExecutionDurationInAllGraphs_ = minMaxExecTime.second;

    minWaitDurationInAllGraphs_ = minMaxWaitTime.first;
    maxWaitDurationInAllGraphs_ = minMaxWaitTime.second;

    // Compute range
    rangeExecutionDurationInAllGraphs_ =
        maxExecutionDurationInAllGraphs_ == minExecutionDurationInAllGraphs_ ?
        std::chrono::nanoseconds(1) :
        maxExecutionDurationInAllGraphs_ - minExecutionDurationInAllGraphs_;

    rangeWaitDurationInAllGraphs_ =
        maxWaitDurationInAllGraphs_ == minWaitDurationInAllGraphs_ ?
        std::chrono::nanoseconds(1) :
        maxWaitDurationInAllGraphs_ - minWaitDurationInAllGraphs_;

    graphTotalExecution_ =
        graph->dequeueExecDuration() == std::chrono::nanoseconds::zero() ?
        std::chrono::system_clock::now() - graph->startExecutionTimeStamp() : graph->dequeueExecDuration();
  }

  /// @brief Dot Printer destructor
  ~DotPrinter() override { outputFile_.close(); }

  /// @brief Print graph header under dot format
  /// @param graph Graph to print
  void printGraphHeader(core::abstraction::GraphNodeAbstraction const *graph) override {
    // If the graph is the outer graph, i.e. the main graph
    if (!graph->isRegistered()) {
      outputFile_
          << "digraph " << graph->id()
          << " {\nlabel=\"" << graph->name();
      if (debugOptions_ == DebugOptions::ALL) { outputFile_ << " " << graph->id(); }

      outputFile_ << "\\nExecution duration:" << durationPrinter(this->graphTotalExecution_)
                  << "\\nCreation duration:" << durationPrinter(graph->graphConstructionDuration())
                  << "\"; fontsize=25; penwidth=5; labelloc=top; labeljust=left; \n";

      // If the graph is an inner graph, i.e. a graph of the outer graph
    } else {
      outputFile_ << "subgraph cluster" << graph->id() << " {\nlabel=\"" << graph->name();
      if (debugOptions_ == DebugOptions::ALL) {
        outputFile_ << " " << graph->id();
      }
      outputFile_
          << "\"; fontsize=25; penwidth=5; fillcolor=\""
          << colorFormatConvertor(graph->printOptions().background())
          << "\";\n";
    }
    outputFile_.flush();
  }

  /// @brief Print graph footer under dot format
  /// @param graph Graph to print
  void printGraphFooter(core::abstraction::GraphNodeAbstraction const *graph) override {

    // Print all edge declarations that has not already been printed
    for (auto &edge : *edges_) {
      if (edge.belongingGraph() == graph) {
        edge.printDeclaration(outputFile_);
      }
    }

    // If the graph is the outer graph
    if (!graph->isRegistered()) {
      // Print all the stored edges

      for (auto const &edge : *(this->edges_)) { edge.printEdges(outputFile_); }
    }
    // Close the dot subgraph
    outputFile_ << "}\n";
    outputFile_.flush();
  }

  /// @brief Print node information depending on the kind of node under the dot format
  /// @param node Node to print
  void printNodeInformation(core::abstraction::NodeAbstraction *node) override {
    // If the node is not a graph
    if (auto task = dynamic_cast<core::abstraction::TaskNodeAbstraction *>(node)) {
      //If all group node to be printed
      if (this->structureOptions_ == StructureOptions::ALL || this->structureOptions_ == StructureOptions::THREADING) {
        // Get and print the node information
        outputFile_ << getTaskInformation(task);
        // If only one node per group need to be printed with gathered information
      } else {
        if (auto copyableNode = dynamic_cast<core::abstraction::AnyGroupableAbstraction const *>(task)) {
          // If the node is the group main node
          if (copyableNode == copyableNode->groupRepresentative()) {
            // Get and print the node information
            outputFile_ << getTaskInformation(task);
          }
        } else {
          // Case for printing state manager while not printing all nodes
          outputFile_ << getTaskInformation(task);
        }
      }
    }
    outputFile_.flush();
  }

  /// @brief Print an edge between node from and to for the type edgetype under the dot format
  /// @param from Sender node
  /// @param to Receiving node
  /// @param edgeType Type of data transmitted through the edge
  /// @param queueSize Number of elements in the receiving queue
  /// @param maxQueueSize Maximum size of the receiving queue
  void printEdge(core::abstraction::NodeAbstraction const *from, core::abstraction::NodeAbstraction const *to,
                 std::string const &edgeType,
                 size_t const &queueSize, size_t const &maxQueueSize) override {

    std::ostringstream oss;
    std::string idToFind, label;

    for (auto &source : from->ids()) {
      for (auto &dest : to->ids()) {
        auto edge = getOrCreateEdge(dest.second, edgeType, to->belongingGraph());

        if (edge->extraLabel().empty()) {
          if (this->structureOptions_ == StructureOptions::QUEUE || this->structureOptions_ == StructureOptions::ALL) {
            oss << "QS=" << queueSize << "\\nMQS=" << maxQueueSize;
            edge->addExtraLabel(oss.str());
            oss.str("");
          }
        }
        edge->addArrival(source.first);
        edge->addExit(dest.first);

        if (this->structureOptions_ == StructureOptions::THREADING ||
            this->structureOptions_ == StructureOptions::ALL) {
          if (auto copyableSender = dynamic_cast<core::abstraction::AnyGroupableAbstraction const *>(from)) {
            for (auto groupMember : *copyableSender->group()) { edge->addArrival(groupMember->nodeId()); }
          }
        }
      }
    }
  }

  /// @brief Print a group of nodes under the dot format
  /// @param representative Group node representative
  /// @param group Group of nodes
  /// @throw std::runtime_error if the group representative node does not derives from AnyGroupableAbstraction
  void printGroup(core::abstraction::NodeAbstraction *representative,
                  std::vector<core::abstraction::NodeAbstraction *> const &group) override {
    bool const printAllGroupMembers =
        this->structureOptions_ == StructureOptions::THREADING || this->structureOptions_ == StructureOptions::ALL;

    auto copyableRepr = dynamic_cast<core::abstraction::AnyGroupableAbstraction *>(representative);
    auto printRepr = dynamic_cast<core::abstraction::PrintableAbstraction *>(representative);
    if (copyableRepr == nullptr || printRepr == nullptr) {
      std::ostringstream oss;
      oss << "Internal error in: " << __FUNCTION__
          << " a group of node should be created with node that derives from AnyGroupableAbstraction and PrintableAbstraction";
      throw std::runtime_error(oss.str());
    }

    // Print header
    // If all group node to be printed
    if (printAllGroupMembers) {
      // Create a dot subgraph for the task group
      outputFile_ << "subgraph cluster" << representative->id()
                  << " {\nlabel=\"\"; penwidth=3; style=filled; fillcolor=\"#ebf0fa\"; color=\"#4e78cf\";\n";

    }

    printRepr->visit(this);

    if (printAllGroupMembers) {
      for (auto groupMember : group) {
        if(auto printGroupMember = dynamic_cast<core::abstraction::PrintableAbstraction *>(groupMember)){
          printGroupMember->visit(this);
        }else {
          std::ostringstream oss;
          oss << "Internal error in: " << __FUNCTION__
              << " a group of node should be created with nodes that derive from AnyGroupableAbstraction and PrintableAbstraction";
          throw std::runtime_error(oss.str());
        }
      }
    }

    if (printAllGroupMembers) {
      outputFile_ << "}\n";
    }

    outputFile_.flush();
  }

  /// @brief Print outer graph source under the dot format
  /// @param source Source of the graph
  void printSource(core::abstraction::NodeAbstraction const *source) override {
    outputFile_ << source->id() << " [label=\"" << source->name();
    if (debugOptions_ == DebugOptions::ALL) {
      outputFile_ << " " << source->id() << " \\(Graph:" << source->belongingGraph()->id() << "\\)";
    }
    outputFile_ << "\", shape=invhouse];\n";
    outputFile_.flush();
  }

  /// @brief Print outer graph sink under the dot format
  /// @param sink Sink of the graph
  void printSink(core::abstraction::NodeAbstraction const *sink) override {
    outputFile_ << sink->id() << " [label=\"" << sink->name();
    if (debugOptions_ == DebugOptions::ALL) {
      outputFile_ << " " << sink->id() << " \\(Graph:" << sink->belongingGraph()->id() << "\\)";
    }
    outputFile_ << "\", shape=point];\n";
    outputFile_.flush();
  }

  /// @brief Print execution pipeline header under the dot format
  /// @param ep Execution pipeline to print
  /// @param switchNode Execution pipeline switch
  void printExecutionPipelineHeader(core::abstraction::ExecutionPipelineNodeAbstraction const *ep,
                                    core::abstraction::NodeAbstraction const *switchNode) override {
    //Print the dot subgraph header
    outputFile_ << "subgraph cluster" << ep->id() << " {\nlabel=\"" << ep->name();
    if (debugOptions_ == DebugOptions::ALL) { outputFile_ << " " << ep->id() << " / " << switchNode->id(); }
    // Print a "triangle" node to represent the execution pipeline switch
    outputFile_ << "\"; penwidth=1; style=dotted; style=filled; fillcolor=\""
                << colorFormatConvertor(ep->printOptions().background())
                << "\";\n "
                << switchNode->id() << "[label=\"\", shape=triangle];\n";
    outputFile_.flush();
  }

  /// @brief Print execution pipeline footer under the dot format
  void printExecutionPipelineFooter() override {
    outputFile_ << "}\n";
    outputFile_.flush();
  }

 private:
  /// @brief Get an existing edge or create a new edge
  /// @param id Edge id
  /// @param type Type transmitted through the edge
  /// @param belongingGraph Graph holding the edge
  /// @return Iterator to the edge
  std::vector<Edge>::iterator getOrCreateEdge(
      std::string const &id, std::string const &type,
      hh::core::abstraction::GraphNodeAbstraction *belongingGraph) {
    std::ostringstream ossId;
    ossId << "edge" << id << type;
    Edge temp(ossId.str(), type, belongingGraph);
    auto const & it = std::find(this->edges_->begin(), this->edges_->end(), temp);
    if (it != this->edges_->end()) { return it; }
    else { return this->edges_->insert(this->edges_->end(), temp); }
  }

  /// @brief Print under the dot format the information for a task
  /// @param task Task to print
  /// @return String containing all the information for a task under the dot format
  std::string getTaskInformation(core::abstraction::TaskNodeAbstraction *task) {
    std::stringstream ss;

    auto const copyableTask = dynamic_cast<core::abstraction::AnyGroupableAbstraction const *>(task);
    auto const slotTask = dynamic_cast<core::abstraction::SlotAbstraction *>(task);
    auto const sm = dynamic_cast<core::abstraction::StateManagerNodeAbstraction const *>(task);

    bool const printAllNodes =
        this->structureOptions_ == StructureOptions::THREADING || this->structureOptions_ == StructureOptions::ALL;

    // Print the name
    ss << task->id() << " [label=\"" << task->name();
    // Print the id (address) in case of debug
    if (debugOptions_ == DebugOptions::ALL) {
      ss << " " << task->id() << " \\(" << task->belongingGraph()->id() << "\\)";
    }

    // If the group has to be presented as a single dot node
    if (!printAllNodes) {
      if (copyableTask && copyableTask->isInGroup()) {
        ss << " x " << copyableTask->numberThreads();
      }
    }
    // If debug information printed
    if (debugOptions_ == DebugOptions::ALL) {
      if (slotTask) {
        // Print number of active input connection
        ss << "\\nActive inputs connection: " << slotTask->nbNotifierConnected();
      }
      // If all nodes in a group need to be printed
      if (printAllNodes) {
        ss << "\\nThread Active?: " << std::boolalpha << task->isActive();
        // If all nodes in a group should NOT be printed
      } else {
        if (copyableTask) {
          ss << "\\nActive threads: " << copyableTask->numberActiveThreadInGroup();
        }
      }
    }
    // If all nodes in a group need to be printed OR is state manager
    if (printAllNodes || sm) {
      if(inputOptions_ == InputOptions::GATHERED) {
        ss << "\\nElements: " << task->numberReceivedElements();
      }else {
        ss << "\\nElements/Input:";
        for(auto const &[typeStr, nbElements] : task->nbElementsPerInput()){
          ss << " (" << typeStr << ")" << nbElements;
        }
      }

      ss << "\\nWait: " << durationPrinter(task->waitDuration());
      if(sm){
        ss << "\\nLock state: " << durationPrinter(sm->acquireStateDuration());
        ss << "\\nEmpty ready list: " << durationPrinter(sm->emptyRdyListDuration());
      }

      if(inputOptions_ == InputOptions::GATHERED){
        ss << "\\nDequeue+Exec: " << durationPrinter(task->dequeueExecDuration());
        ss << "\\nExec/Element: " << durationPrinter(task->averageExecutionDurationPerElement());
      }else {
        ss << "\\nDequeue+Exec/Input: ";
        for(auto const &[typeStr, duration] : task->dequeueExecutionDurationPerInput()){
          ss << " (" << typeStr << ") " << durationPrinter(duration);
        }
        ss << "\\nExec/Element/Input:";
        for(auto const &[typeStr, duration] : task->averageExecutionDurationPerInputType()){
          ss << " (" << typeStr << ") " << durationPrinter(duration);
        }
      }

      if (task->hasMemoryManagerAttached()) {
        ss << "\\nMemory manager (" << task->memoryManager()->managedType() << "): "
           << task->memoryManager()->currentSize() << "/" << task->memoryManager()->capacity();
        ss << "\\nMemory Wait: " << durationPrinter(task->memoryWaitDuration());
      }

      // If all nodes in a group should NOT be printed
    } else {
      //Get the time in the groups
      if (copyableTask) {
        // Print the number of element received per task

        if (copyableTask->numberThreads() > 1) {
          if(inputOptions_ == InputOptions::GATHERED) {
            ss << "\\nElements: ";
            auto minmaxElements = copyableTask->minmaxNumberElementsReceivedGroup();
            auto meanSDNumberElements = copyableTask->meanSDNumberElementsReceivedGroup();
            ss << "Min: " << minmaxElements.first << ""
               << " / Avg: " << std::setw(1) << meanSDNumberElements.first
               << " +- " << std::setw(1) << meanSDNumberElements.second
               << " / Max: " << minmaxElements.second;
          }else {
            ss << "\\nElements/Input: ";
            auto minmaxElementsPerInputs = copyableTask->minmaxNumberElementsReceivedGroupPerInput();
            auto meanSDNumberElementsPerInputs = copyableTask->meanSDNumberElementsReceivedGroupPerInput();
            for(const auto& [key, minMax] : minmaxElementsPerInputs){
              ss << "\\n(" << key << ") Min: " << minMax.first
                 << " Avg: " << std::setw(1) << meanSDNumberElementsPerInputs.at(key).first
                 << " +- " << std::setw(1) << meanSDNumberElementsPerInputs.at(key).second
                 << " Max: " << minMax.second;
            }
          }
        } else {
          if(inputOptions_ == InputOptions::GATHERED) {
            ss << "\\nElements: ";
            ss << task->numberReceivedElements();
          }else {
            ss << "\\nElements/Input:";
            for(const auto& [key, nbElem] : task->nbElementsPerInput()){
              ss << " (" << key << ") " << nbElem;
            }
          }
        }
        // Print the wait time
        ss << "\nWait Time: ";
        if (copyableTask->numberThreads() > 1) {
          auto minmaxWait = copyableTask->minmaxWaitDurationGroup();
          auto meanSDWait = copyableTask->meanSDWaitDurationGroup();
          ss << "Min: " << durationPrinter(minmaxWait.first)
             << " / Avg: " << durationPrinter(meanSDWait.first)
             << " +- " << durationPrinter(meanSDWait.second)
             << " / Max: " << durationPrinter(minmaxWait.second) << "\\n";
        } else { ss << durationPrinter(task->waitDuration()) << "\\n"; }
        // Print the execution time
        if(inputOptions_ == InputOptions::GATHERED) {
          ss << "Dequeue+Exec: ";
          if (copyableTask->numberThreads() > 1) {
            auto minmaxExec = copyableTask->minmaxDequeueExecutionDurationGroup();
            auto meanSDExec = copyableTask->meanSDDequeueExecutionDurationGroup();
            ss << "Min: " << durationPrinter(minmaxExec.first)
               << " / Avg: " << durationPrinter(meanSDExec.first)
               << " +- " << durationPrinter(meanSDExec.second)
               << " / Max: " << durationPrinter(minmaxExec.second) << "\\n";
          } else { ss << durationPrinter(task->dequeueExecDuration()) << "\\n"; }
        }else {
          ss << "Dequeue+Exec/Input:";
          if (copyableTask->numberThreads() > 1) {
            auto minmaxExec = copyableTask->minmaxDequeueExecutionDurationGroupPerInput();
            auto meanSDExec = copyableTask->meanSDDequeueExecutionDurationGroupPerInput();
            for(auto const & [key, minMax] : minmaxExec) {
              ss << "\\n(" << key << ") Min: " << durationPrinter(minmaxExec.at(key).first)
                 << " / Avg: " << durationPrinter(meanSDExec.at(key).first)
                 << " +- " << durationPrinter(meanSDExec.at(key).second)
                 << " / Max: " << durationPrinter(minmaxExec.at(key).second);
            }
            ss << "\\n";
          } else {
            for(auto const & [key, duration] : task->dequeueExecutionDurationPerInput()){
              ss << " (" << key << ") " << durationPrinter(duration);
            }
            ss << "\\n";
          }

        }
        // Print the execution time per Element
        if(inputOptions_ == InputOptions::GATHERED) {
          ss << "Exec: ";
          if (copyableTask->numberThreads() > 1) {
            auto minmaxExecPerElement = copyableTask->minmaxExecTimePerElementGroup();
            auto meanSDExecPerElement = copyableTask->meanSDExecTimePerElementGroup();
            ss
               << "  Min: " << durationPrinter(minmaxExecPerElement.first)
               << " / Avg: " << durationPrinter(meanSDExecPerElement.first) << " +- "
               << durationPrinter(meanSDExecPerElement.second)
               << " / Max: " << durationPrinter(minmaxExecPerElement.second) << "\\n";
          } else { ss << durationPrinter(task->averageExecutionDurationPerElement()) << "\\n"; }
        }else {
          ss << "Exec/input: ";
          if (copyableTask->numberThreads() > 1) {
            auto minMaxExec = copyableTask->minmaxExecTimePerElementGroupPerInput();
            auto meanSDExec = copyableTask->meanSDExecTimePerElementGroupPerInput();
            for(auto const & [key, minMax] : minMaxExec) {
              ss << "\\n(" << key << ") Min: " << durationPrinter(minMaxExec.at(key).first)
                 << " / Avg: " << durationPrinter(meanSDExec.at(key).first)
                 << " +- " << durationPrinter(meanSDExec.at(key).second)
                 << " / Max: " << durationPrinter(minMaxExec.at(key).second);
            }
            ss << "\\n";
          } else {
            for(auto const & [key, duration] : task->executionDurationPerInput()){
              ss << " (" << key << ") " << durationPrinter(duration);
            }
            ss << "\\n";
          }
        }

        // Print the memory wait time
        if (task->hasMemoryManagerAttached()) {
          ss << "Memory manager (" << task->memoryManager()->managedType() << "): "
             << task->memoryManager()->currentSize() << "/" << task->memoryManager()->capacity();
          ss << "\\nMemory Wait Time: ";
          if (copyableTask->numberThreads() == 1) {
            ss << durationPrinter(task->memoryWaitDuration()) << "\\n";
          } else {
            auto minmaxWaitMemory = copyableTask->minmaxMemoryWaitTimeGroup();
            auto meanSDWaitMemory = copyableTask->meanSDMemoryWaitTimePerElementGroup();
            ss
               << "  Min: " << durationPrinter(minmaxWaitMemory.first) << "\\n"
               << "  Avg: " << durationPrinter(meanSDWaitMemory.first) << " +-"
               << durationPrinter(meanSDWaitMemory.second) << "\\n"
               << "  Max: " << durationPrinter(minmaxWaitMemory.second) << "\\n";
          }
        }
      }
    }
    // If extra information has been defined by the user, print it
    auto extraInfo = task->extraPrintingInformation();
    if (!extraInfo.empty()) { ss << "\\n" << extraInfo; }

    ss << "\"";
    ss << ",shape=rect";
    // Change the color of the rect depending on the user choice
    switch (this->colorScheme_) {
      case ColorScheme::EXECUTION:ss << ",color=" << this->getExecRGB(task->dequeueExecDuration()) << ", penwidth=3";
        break;
      case ColorScheme::WAIT:ss << ",color=" << this->getWaitRGB(task->waitDuration()) << ", penwidth=3";
        break;
      default:break;
    }

    ss << R"(, style=filled, fillcolor=")"
       << colorFormatConvertor(task->printOptions().background())
       << R"(", fontcolor=")"
       << colorFormatConvertor(task->printOptions().font())
       << "\"];\n";
    return ss.str();
  }

  /// @brief Test the path given by the user, print extra information if verbose is used
  /// @param dotFilePath Path to test
  /// @param verbose Verbose option
  /// @throw std::runtime_error if the dotFilePath is not valid (do not represent a file and parent path does not exist)
  void testPath(std::filesystem::path const &dotFilePath, bool verbose) {
    auto directoryPath = dotFilePath.parent_path();
    std::ostringstream oss;
    if (!dotFilePath.has_filename()) {
      oss << "The path: " << dotFilePath << " does not represent a file.";
      throw (std::runtime_error(oss.str()));
    }
    if (!std::filesystem::exists(directoryPath)) {
      oss << "The file " << dotFilePath.filename() << " can not be store in " << directoryPath
          << " because the directory does not  exist.";
      throw (std::runtime_error(oss.str()));
    }
    if (!std::filesystem::exists(directoryPath)) {
      oss << "The file " << dotFilePath.filename() << " can not be store in " << directoryPath
          << " because the directory does not  exist.";
      throw (std::runtime_error(oss.str()));
    }
    if (verbose) {
      if (std::filesystem::exists(dotFilePath)) {
        std::cout << "The file " << dotFilePath.filename() << " will be overwritten." << std::endl;
      } else {
        std::cout << "The file " << dotFilePath.filename() << " will be created." << std::endl;
      }
    }
    outputFile_ = std::ofstream(dotFilePath);
  }

  /// @brief Get the rgb color for the execution time value
  /// @param ns Execution value to get the RGB color
  /// @return RGB color for val
  std::string getExecRGB(std::chrono::nanoseconds const &ns) {
    return colorPicker_
        ->getRGBFromRange(ns, this->minExecutionDurationInAllGraphs_, this->rangeExecutionDurationInAllGraphs_);
  }

  /// @brief Get the rgb color for the wait time value
  /// @param ns Execution value to get the RGB color
  /// @return RGB color for val
  std::string getWaitRGB(std::chrono::nanoseconds const &ns) {
    return colorPicker_->getRGBFromRange(ns, this->minWaitDurationInAllGraphs_, this->rangeWaitDurationInAllGraphs_);
  }

  /// @brief Print a duration with the right precision
  /// @param ns duration in ns
  /// @return String representing a duration with the right format
  static std::string durationPrinter(std::chrono::nanoseconds const &ns) {
    std::ostringstream oss;

    // Cast with precision loss
    auto s = std::chrono::duration_cast<std::chrono::seconds>(ns);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(ns);
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(ns);

    if (s > std::chrono::seconds::zero()) {
      oss << s.count() << "." << std::setfill('0') << std::setw(3) << (ms - s).count() << "s";
    } else if (ms > std::chrono::milliseconds::zero()) {
      oss << ms.count() << "." << std::setfill('0') << std::setw(3) << (us - ms).count() << "ms";
    } else if (us > std::chrono::microseconds::zero()) {
      oss << us.count() << "." << std::setfill('0') << std::setw(3) << (ns - us).count() << "us";
    } else {
      oss << ns.count() << "ns";
    }
    return oss.str();
  }

  /// @brief Print a color into a format understood by Graphiz
  /// @param color Color definition
  /// @return String containing the color representation
  static std::string colorFormatConvertor(hh::tool::PrintOptions::Color const &color) {
    std::ostringstream ss;
    ss << "#"
       << std::setw(2) << std::setfill('0') << std::hex << (int) color.r_
       << std::setw(2) << std::setfill('0') << std::hex << (int) color.g_
       << std::setw(2) << std::setfill('0') << std::hex << (int) color.b_
       << std::setw(2) << std::setfill('0') << std::hex << (int) color.a_;

    return ss.str();
  }
};

}

#endif //HEDGEHOG_DOT_PRINTER_H
