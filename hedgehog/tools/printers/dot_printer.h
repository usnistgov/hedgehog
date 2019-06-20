//
// Created by anb22 on 3/13/19.
//

#ifndef HEDGEHOG_DOT_PRINTER_H
#define HEDGEHOG_DOT_PRINTER_H

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

#include "abstract_printer.h"

enum class ColorScheme { NONE, EXECUTION, WAIT };
enum class StructureOptions { NONE, ALLTHREADING, QUEUE, ALL };
enum class DebugOptions { NONE, DEBUG };

class DotPrinter : public AbstractPrinter {
 private:
  std::vector<std::string> edges_;
  std::ofstream outputFile_;
  std::set<CoreNode const *> uniqueNodes_ = {};
  ColorScheme colorScheme_;
  StructureOptions structureOptions_;
  DebugOptions debugOptions_;

  uint64_t
      maxExecutionTime_ = {},
      minExecutionTime_ = {},
      rangeExecutionTime_ = {},
      maxWaitTime_ = {},
      minWaitTime_ = {},
      rangeWaitTime_ = {},
      graphExecutionDuration_ = {};

 public:
  explicit DotPrinter(std::filesystem::path const &dotFilePath,
                      ColorScheme colorScheme,
                      StructureOptions structureOptions,
                      DebugOptions debugOptions,
                      CoreNode *graph)
      : AbstractPrinter(),
        edges_({}),
        uniqueNodes_({}),
        colorScheme_(colorScheme),
        structureOptions_(structureOptions),
        debugOptions_(debugOptions) {
    assert(graph != nullptr);
    auto directoryPath = dotFilePath.parent_path();
    if (dotFilePath.has_filename()) {
      if (std::filesystem::exists(directoryPath)) {
        if (std::filesystem::exists(dotFilePath)) {
          std::cout
              << "The file " << dotFilePath.filename() << " will be overwritten." << std::endl;
        }
        outputFile_ = std::ofstream(dotFilePath);

      } else {

        HLOG(0,
             "The file " << dotFilePath.filename() << " can not be store in " << directoryPath
                         << " because the directory does not  exist.")
        exit(42);
      }
    } else {
      HLOG(0, "The path: " << dotFilePath << " does not represent a file.")
      exit(42);
    }
    minExecutionTime_ = graph->minExecutionTime().count();
    maxExecutionTime_ = graph->maxExecutionTime().count();
    minWaitTime_ = graph->minWaitTime().count();
    maxWaitTime_ = graph->maxWaitTime().count();
    rangeExecutionTime_ = maxExecutionTime_ - minExecutionTime_ == 0 ? 1 : maxExecutionTime_ - minExecutionTime_;
    rangeWaitTime_ = maxWaitTime_ - minWaitTime_ == 0 ? 1 : maxWaitTime_ - minWaitTime_;
    graphExecutionDuration_ =
        graph->executionDuration().count() == 0 ?
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - graph->startExecutionTimeStamp()).count()
                                                : graph->executionDuration().count();
  }

  ~DotPrinter() override {
    outputFile_.close();
  }

  void printNodeInformation(CoreNode *node) final {
    if (this->structureOptions_ == StructureOptions::ALL || this->structureOptions_ == StructureOptions::ALLTHREADING) {
      if (node->type() != NodeType::Graph) {
        outputFile_ << getNodeInformation(node);
      }
    } else {
      if (node->type() != NodeType::Graph && node->id() == node->coreClusterNode()->id()) {
        outputFile_ << getNodeInformation(node);
      }
    }
    outputFile_.flush();
  }

  bool hasNotBeenVisited(CoreNode const *node) final {
    return uniqueNodes_.insert(node).second;
  }

  void printGraphHeader(CoreNode const *node) final {
    if (!node->isInside()) {
      outputFile_
          << "digraph " << node->id()
          << " {\nlabel=\"" << node->name() << " " << node->id()
          << "\\nExecution time:" << this->durationPrinter(this->graphExecutionDuration_)
          << "\\nCreation time:" << this->durationPrinter(node->creationDuration().count())
          << "\"; fontsize=25; penwidth=5; ranksep=0; labelloc=top; labeljust=left;\n";
    } else {
      outputFile_ << "subgraph cluster" << node->id() << " {\nlabel=\"" << node->name() << " " << node->id()
                  << "\"; fontsize=25; penwidth=5;\n";
    }
    outputFile_.flush();
  }

  void printClusterHeader(CoreNode const *clusterNode) final {
    if (this->structureOptions_ == StructureOptions::ALLTHREADING || this->structureOptions_ == StructureOptions::ALL) {
      outputFile_ << "subgraph cluster" << clusterNode->id() << " {\nlabel=\"\"; penwidth=1; style=dotted;\n";
      outputFile_ << "box" << clusterNode->id() << "[label=\"\", shape=egg];\n";
      outputFile_.flush();
    }
  }

  void printClusterFooter() final {

    if (this->structureOptions_ == StructureOptions::ALLTHREADING || this->structureOptions_ == StructureOptions::ALL) {
      outputFile_ << "}\n";
      outputFile_.flush();
    }
  }

  void printGraphFooter(CoreNode const *node) final {
    if (!node->isInside()) {
      for (std::string const &edge: edges_) {
        outputFile_ << edge << "\n";
      }
    }
    outputFile_ << "}\n";
    outputFile_.flush();
  }

  void printClusterEdge(CoreNode const *clusterNode) final {
    if (this->structureOptions_ == StructureOptions::ALLTHREADING || this->structureOptions_ == StructureOptions::ALL) {
      std::stringstream ss;
      ss << "box" << clusterNode->coreClusterNode()->id() << " -> " << clusterNode->id();
      edges_.push_back(ss.str());
    }
  }

  void printExecutionPipelineHeader(std::string_view const &executionPipelineName,
                                    std::string const &executionPipelineId,
                                    std::string const &switchId) override {
    outputFile_ << "subgraph cluster" << executionPipelineId << " {\n"
                << "label=\"" << executionPipelineName
                << "\"; penwidth=1; style=dotted; style=filled; fillcolor=gray80;\n "
                << switchId << "[label=\"\", shape=triangle];\n";
    outputFile_.flush();
  }

  void printExecutionPipelineFooter() override {
    outputFile_ << "}\n";
    outputFile_.flush();
  }

  void printEdgeSwitchGraphs(CoreNode *to,
                             std::string const &idSwitch,
                             std::string_view const &edgeType) override {
    std::ostringstream
        oss;

    std::string
        idDest,
        headLabel;

    if (this->structureOptions_ == StructureOptions::ALLTHREADING || this->structureOptions_ == StructureOptions::ALL) {
      if (to->isInCluster()) {
        for (auto &dest: to->ids()) {

          idDest = "box" + dest.second;
          oss << idSwitch << " -> " << idDest << "[label=\"" << edgeType << "\"" << "];";
        }
      } else {
        oss << idSwitch << " -> " << to->id() << "[label=\"" << edgeType << "\"" << "];";
      }
    } else if (to->id() == to->coreClusterNode()->id()) {
      oss << idSwitch << " -> " << to->id() << "[label=\"" << edgeType << "\"];";
    }
    edges_.push_back(oss.str());
  }

  void printEdge(CoreNode const *from,
                 CoreNode const *to,
                 std::string_view const &edgeType,
                 size_t const &queueSize,
                 size_t const &maxQueueSize) final {
    std::ostringstream
        oss;

    std::string
        headLabel,
        tailLabel,
        idDest,
        queueStr;

    if (this->structureOptions_ == StructureOptions::QUEUE || this->structureOptions_ == StructureOptions::ALL) {
      oss << " " << queueSize << " (" << maxQueueSize << ")";
      queueStr = oss.str();
      oss.str(std::string());
    }

    if (this->structureOptions_ == StructureOptions::ALLTHREADING || this->structureOptions_ == StructureOptions::ALL) {
      if (from->isInCluster()) {
        for (auto &source: from->ids()) {
          headLabel = ",ltail=cluster" + source.second;
          if (to->isInCluster()) {
            for (auto &dest : to->ids()) {
              tailLabel = ",lhead=cluster" + dest.second;
              idDest = "box" + dest.second;
              oss << source.first << " -> " << idDest << "[label=\"" << edgeType << queueStr << "\"" << headLabel
                  << tailLabel
                  << "];";
            }
          } else {
            oss << source.first << " -> " << to->id() << "[label=\"" << edgeType << queueStr << "\"" << headLabel
                << "];";
          }
        }
      } else {
        if (to->isInCluster()) {
          for (auto &dest : to->ids()) {
            tailLabel = ",lhead=cluster" + dest.second;
            idDest = "box" + dest.second;
            oss << from->id() << " -> " << idDest << "[label=\"" << edgeType << queueStr << "\"" << headLabel
                << tailLabel << "];";
          }
        } else {
          oss << from->id() << " -> " << to->id() << "[label=\"" << edgeType << queueStr << "\"" << headLabel
              << "];";
        }
      }
      edges_.push_back(oss.str());
    } else if (from->id() == from->coreClusterNode()->id() && to->id() == to->coreClusterNode()->id()) {
      oss << from->id() << " -> " << to->id() << "[label=\"" << edgeType << queueStr << "\"];";
      edges_.push_back(oss.str());
    }
  }

 private:
  std::string getNodeInformation(CoreNode *node) {
    std::stringstream ss;

    ss << node->id() << " [label=\"" << node->name() << " " << node->id() << " \\(" << node->threadId() << ", "
       << node->graphId() << "\\)";

    switch (node->type()) {
      case NodeType::Graph:ss << "\"";
        break;
      case NodeType::Source:ss << "\", shape=doublecircle";
        break;
      case NodeType::Task:
        if (!(this->structureOptions_ == StructureOptions::ALLTHREADING
            || this->structureOptions_ == StructureOptions::ALL)) {
          ss << " x " << node->numberThreads();
        }
        if (debugOptions_ == DebugOptions::DEBUG) {
          ss << "\\nActive input connection: " << dynamic_cast<CoreSlot *>(node)->numberInputNodes();
          if (this->structureOptions_ == StructureOptions::ALLTHREADING
              || this->structureOptions_ == StructureOptions::ALL) {
            ss << "\\nThread Active?: " << std::boolalpha << dynamic_cast<CoreSlot *>(node)->isActive();
          } else {
            ss << "\\nActive threads: " << dynamic_cast<CoreSlot *>(node)->numberActiveThreadInCluster();
          }
        }
        if (this->structureOptions_ == StructureOptions::ALLTHREADING
            || this->structureOptions_ == StructureOptions::ALL) {
          ss << "\\nWait Time: " << this->durationPrinter(node->waitTime().count());
          ss << "\\nExecution Time: " << this->durationPrinter(node->executionTime().count());
        } else {
          ss << "\\nWait Time: " << this->durationPrinter(node->meanWaitTimeCluster().count()) << " +- "
             << this->durationPrinter(node->stdvWaitTimeCluster());
          ss << "\\nExecution Time: " << this->durationPrinter(node->meanExecTimeCluster().count()) << " +- "
             << this->durationPrinter(node->stdvExecTimeCluster());
        }
        if (node->extraPrintingInformation() != "") {
          ss << "\\nExtra Information: " << node->extraPrintingInformation();
        }
        ss << "\"";
        ss << ",shape=circle";
        switch (this->colorScheme_) {
          case ColorScheme::EXECUTION:ss << ",color=" << this->getExecRGB(node->executionTime().count());
            break;
          case ColorScheme::WAIT:ss << ",color=" << this->getWaitRGB(node->waitTime().count());
            break;
          default:break;
        }
        break;
      case NodeType::StateManager:
        if (debugOptions_ == DebugOptions::DEBUG) {
          ss << "\\nActive input connection: " << dynamic_cast<CoreSlot *>(node)->numberInputNodes();
          ss << "\\nActive threads: " << dynamic_cast<CoreSlot *>(node)->numberActiveThreadInCluster();
        }
        ss << "\\nWait Time: " << this->durationPrinter(node->waitTime().count());
        ss << "\\nExecution Time: " << this->durationPrinter(node->executionTime().count());
        ss << "\"";
        ss << ",shape=diamond";
        switch (this->colorScheme_) {
          case ColorScheme::EXECUTION:ss << ",color=" << this->getExecRGB(node->executionTime().count());
            break;
          case ColorScheme::WAIT:ss << ",color=" << this->getWaitRGB(node->waitTime().count());
            break;
          default:break;
        }
        break;
      case NodeType::Sink:ss << "\",shape=point";
        break;
      default:break;
    }
    ss << "];\n";

    return ss.str();
  }

  std::string getRGBFromRange(uint64_t const &val, uint64_t const &min, uint64_t const &range) {
    uint64_t posRedToBlue = (val - min) * 255 / range;
    posRedToBlue = posRedToBlue > 255 ? 255 : posRedToBlue;
    std::stringstream ss;
    ss << "\"#"
       << std::setfill('0') << std::setw(2) << std::hex << posRedToBlue
       << "00"
       << std::setfill('0') << std::setw(2) << std::hex << 255 - posRedToBlue
       << "\"";
    HLOG(0, "PRINT RGB RANGE " << val << " " << min << " " << range << " " << posRedToBlue)
    return ss.str();
  }

  std::string getExecRGB(uint64_t val) {
    return getRGBFromRange(val, this->minExecutionTime_, this->rangeExecutionTime_);
  }

  std::string getWaitRGB(uint64_t val) {
    return getRGBFromRange(val, this->minWaitTime_, this->rangeWaitTime_);
  }

  std::string durationPrinter(uint64_t duration) {
    std::ostringstream oss;
    uint64_t
        s = (duration % 1000000000) / 1000000,
        mS = (duration % 1000000) / 1000,
        uS = (duration % 1000);

    if (s > 0) {
      oss << s << "." << std::setfill('0') << std::setw(3) << mS << "s";
    } else if (mS > 0) {
      oss << mS << "." << std::setfill('0') << std::setw(3) << uS << "ms";
    } else {
      oss << duration << "us";
    }
    return oss.str();
  }
};

#endif //HEDGEHOG_DOT_PRINTER_H