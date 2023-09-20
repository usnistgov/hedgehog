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



#ifndef HEDGEHOG_ANY_COPYABLE_ABSTRACTION_H
#define HEDGEHOG_ANY_COPYABLE_ABSTRACTION_H

#include <cstddef>
#include <numeric>
#include <cmath>
#include <algorithm>

#include "node/task_node_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog abstraction namespace
namespace abstraction {

/// @brief Abstraction for cores/nodes that can form groups
class AnyGroupableAbstraction {
 private:
  size_t const numberThreads_ = 0; ///< Number of threads
  AnyGroupableAbstraction *groupRepresentative_ = nullptr; ///< Group representative
  std::shared_ptr<std::set<AnyGroupableAbstraction *>> group_ = nullptr; ///< Node's group

 public:
  /// @brief Constructor using the number of threads
  /// @param numberThreads Number of threads (cores) in the group
  explicit AnyGroupableAbstraction(size_t const numberThreads) :
      numberThreads_(numberThreads),
      groupRepresentative_(this),
      group_(std::make_shared<std::set<AnyGroupableAbstraction *>>()) {
    group_->insert(this);
  }

  /// @brief Default destructor
  virtual ~AnyGroupableAbstraction() = default;

  /// @brief Accessor to the number of threads
  /// @return  Group's number of threads
  [[nodiscard]] size_t numberThreads() const { return numberThreads_; }

  /// @brief Test if a group is needed
  /// @return True if numberThreads != 1, else false
  [[nodiscard]] bool isInGroup() const { return numberThreads_ != 1; }

  /// @brief Accessor to the node id (redirection to NodeAbstraction::nodeId)
  /// @return Node id
  [[nodiscard]] std::string nodeId() const {
    if (auto thisAsNode = dynamic_cast<NodeAbstraction const *>(this)) {
      return thisAsNode->id();
    } else {
      throw std::runtime_error("A group representative should be a NodeAbstraction.");
    };
  }

  /// @brief Group of cores accessor
  /// @return Group of cores
  [[nodiscard]] std::shared_ptr<std::set<AnyGroupableAbstraction *>> const &group() const {
    return group_;
  }

  /// @brief Create a set of the NodeAbstraction constituting the group
  /// @return Set of the NodeAbstraction constituting the group
  [[nodiscard]] std::shared_ptr<std::set<NodeAbstraction *>> groupAsNodes() const {
    auto ret = std::make_shared<std::set<NodeAbstraction *>>();
    for (auto copyable : *group_) {
      if (auto node = dynamic_cast<NodeAbstraction *>(copyable)) { ret->insert(node); }
    }
    return ret;
  }

  /// @brief Group representative accessor
  /// @return Group representative
  [[nodiscard]] AnyGroupableAbstraction *groupRepresentative() const { return groupRepresentative_; }

  /// @brief Group id representative accessor
  /// @return Group id representative
  /// @throw std::runtime_error A group representative is not a NodeAbstraction
  [[nodiscard]] std::string groupRepresentativeId() const {
    if (auto representativeAsNode = dynamic_cast<NodeAbstraction const *>(groupRepresentative_)) {
      return representativeAsNode->id();
    } else {
      throw std::runtime_error("A group representative should be a NodeAbstraction.");
    };
  }

  /// @brief Group representative setter
  /// @param groupRepresentative Group representative to set
  void groupRepresentative(AnyGroupableAbstraction *groupRepresentative) { groupRepresentative_ = groupRepresentative; }

  /// @brief Group setter
  /// @param group Group to set
  void group(std::shared_ptr<std::set<AnyGroupableAbstraction *>> const &group) { group_ = group; }

  /// @brief Accessor to the number of nodes alive in the group
  /// @return Number of nodes alive in the group
  /// @throw std::runtime_error The cores in the groups are not of the same type
  [[nodiscard]] size_t numberActiveThreadInGroup() const {
    size_t count = 0;
    if (dynamic_cast<TaskNodeAbstraction const *>(this)) {
      count = (size_t) std::count_if(
          this->group_->cbegin(), this->group_->cend(),
          [](auto nodeInGroup) {
            if (auto taskGroup = dynamic_cast<TaskNodeAbstraction const *>(nodeInGroup)) {
              return taskGroup->isActive();
            } else {
              throw std::runtime_error("All the nodes in a group should be of the same type: the representative is a "
                                       "TaskNodeAbstraction, but not the nodes in its group.");
            }
          }
      );
    };

    return count;
  }

  /// @brief Accessor to the min / max wait duration of the nodes in the group
  /// @return Min / max wait duration in nanoseconds of the nodes in the group
  /// @throw std::runtime_error All the nodes in a group are not of the same types
  [[nodiscard]] std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> minmaxWaitDurationGroup() const {
    std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> minMax =
        {std::chrono::nanoseconds::zero(), std::chrono::nanoseconds::zero()};

    if (dynamic_cast<TaskNodeAbstraction const *>(this)) {
      auto minMaxElem = std::minmax_element(
          this->group_->cbegin(), this->group_->cend(),
          [](auto lhs, auto rhs) {
            auto lhsAsTask = dynamic_cast<TaskNodeAbstraction const *>(lhs);
            auto rhsAsTask = dynamic_cast<TaskNodeAbstraction const *>(rhs);
            if (lhsAsTask && rhsAsTask) {
              return lhsAsTask->waitDuration() < rhsAsTask->waitDuration();
            } else {
              throw std::runtime_error("All the nodes in a group should be of the same type, the representative is of"
                                       " type TaskNodeAbstraction but not the nodes in its group.");
            }
          }
      );

      auto minTask = dynamic_cast<TaskNodeAbstraction *>(*minMaxElem.first);
      auto maxTask = dynamic_cast<TaskNodeAbstraction *>(*minMaxElem.second);
      if (minTask && maxTask) {
        minMax = {minTask->waitDuration(), maxTask->waitDuration()};
      } else {
        throw std::runtime_error("All the nodes in a group should be of the same type, the representative is of"
                                 " type TaskNodeAbstraction but not the nodes in its group.");
      }
    }

    return minMax;
  }

  /// @brief Accessor to the mean / standard deviation wait duration of the nodes in the group
  /// @return mean / standard deviation  wait duration in nanoseconds of the nodes in the group
  /// @throw std::runtime_error All the nodes in a group are not of the same types
  [[nodiscard]] std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> meanSDWaitDurationGroup() const {
    std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> meanSD =
        {std::chrono::nanoseconds::zero(), std::chrono::nanoseconds::zero()};
    if (dynamic_cast<TaskNodeAbstraction const *>(this)) {
      std::chrono::nanoseconds
          sum = std::chrono::nanoseconds::zero(),
          mean = std::chrono::nanoseconds::zero();
      double
          sd = 0;

      for (auto taskInGroup : *this->group_) {
        auto task = dynamic_cast<TaskNodeAbstraction *>(taskInGroup);
        if (task) {
          sum += task->waitDuration();
        } else {
          throw std::runtime_error("All nodes in a group should be of the same type, the representative derives from "
                                   "TaskNodeAbstraction but not the group members.");
        }
      }
      mean = sum / (this->group_->size());

      for (auto taskInGroup : *this->group_) {
        auto task = dynamic_cast<TaskNodeAbstraction *>(taskInGroup);
        if (task) {
          auto diff = (double) (task->waitDuration().count() - mean.count());
          sd += diff * diff;
        } else {
          throw std::runtime_error("All nodes in a group should be of the same type, the representative derives from "
                                   "TaskNodeAbstraction but not the group members.");
        }
      }
      meanSD = {
          mean,
          std::chrono::nanoseconds((int64_t) std::sqrt(sd / (double) this->group_->size()))};
    }
    return meanSD;
  }

  /// @brief Accessor to the min / max dequeue + execution duration of the nodes in the group
  /// @return Min / max dequeue + execution duration in nanoseconds of the nodes in the group
  [[nodiscard]] std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds>
  minmaxDequeueExecutionDurationGroup() const {
    auto groupAsNode = this->groupAsNodes();
    auto minMaxElem = std::minmax_element(
        groupAsNode->cbegin(), groupAsNode->cend(),
        [](auto lhs, auto rhs) {
          return lhs->dequeueExecDuration() < rhs->dequeueExecDuration();
        }
    );
    return {(*minMaxElem.first)->dequeueExecDuration(),
            (*minMaxElem.second)->dequeueExecDuration()};
  }

  /// @brief Accessor to the mean / standard deviation dequeue + execution duration of the nodes in the group
  /// @return Mean / standard deviation dequeue + execution duration in nanoseconds of the nodes in the group
  [[nodiscard]] std::pair<std::chrono::nanoseconds,
                          std::chrono::nanoseconds> meanSDDequeueExecutionDurationGroup() const {
    auto groupAsNode = groupAsNodes();
    std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> meanSD =
        {std::chrono::nanoseconds::zero(), std::chrono::nanoseconds::zero()};

    std::chrono::nanoseconds
        sum = std::chrono::nanoseconds::zero(),
        mean = std::chrono::nanoseconds::zero();
    double
        sd = 0;

    for (auto taskInGroup : *groupAsNode) {
      sum += taskInGroup->dequeueExecDuration();
    }
    mean = sum / (groupAsNode->size());

    for (auto taskInGroup : *groupAsNode) {
      sd += (double) (taskInGroup->dequeueExecDuration().count() - mean.count()) *
          (double) (taskInGroup->dequeueExecDuration().count() - mean.count());
    }

    meanSD = {mean, std::chrono::nanoseconds((int64_t) std::sqrt(sd / (double) groupAsNode->size()))};

    return meanSD;
  }

  /// @brief Accessor to the min / max execution per elements duration of the nodes in the group
  /// @return Min / max execution per elements duration in nanoseconds of the nodes in the group
  /// @throw std::runtime_error All the nodes in a group are not of the same types
  [[nodiscard]] std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> minmaxExecTimePerElementGroup() const {
    std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> minMax =
        {std::chrono::nanoseconds::zero(), std::chrono::nanoseconds::zero()};

    if (dynamic_cast<TaskNodeAbstraction const *>(this)) {
      auto minMaxElem = std::minmax_element(
          this->group_->cbegin(), this->group_->cend(),
          [](auto lhs, auto rhs) {
            auto lhsAsTask = dynamic_cast<TaskNodeAbstraction const *>(lhs);
            auto rhsAsTask = dynamic_cast<TaskNodeAbstraction const *>(rhs);
            if (lhsAsTask && rhsAsTask) {
              return lhsAsTask->averageExecutionDurationPerElement() < rhsAsTask->averageExecutionDurationPerElement();
            } else {
              throw std::runtime_error("All the nodes in a group should be of the same type, the representative is of"
                                       " type TaskNodeAbstraction but not the group members.");
            }
          }
      );

      auto minTask = dynamic_cast<TaskNodeAbstraction *>(*minMaxElem.first);
      auto maxTask = dynamic_cast<TaskNodeAbstraction *>(*minMaxElem.second);
      if (minTask && maxTask) {
        minMax = {minTask->averageExecutionDurationPerElement(), maxTask->averageExecutionDurationPerElement()};
      } else {
        throw std::runtime_error("All the nodes in a group should be of the same type, the representative is of"
                                 " type TaskNodeAbstraction but not the group members.");
      }
    }
    return minMax;
  }

  /// @brief Accessor to the mean / standard deviation execution per elements duration of the nodes in the group
  /// @return >ean / standard deviation execution per elements duration in nanoseconds of the nodes in the group
  /// @throw std::runtime_error All the nodes in a group are not of the same types
  [[nodiscard]] std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> meanSDExecTimePerElementGroup() const {
    std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> meanSD =
        {std::chrono::nanoseconds::zero(), std::chrono::nanoseconds::zero()};
    if (dynamic_cast<TaskNodeAbstraction const *>(this)) {
      std::chrono::nanoseconds
          sum = std::chrono::nanoseconds::zero(),
          mean = std::chrono::nanoseconds::zero();
      double
          sd = 0;

      for (auto taskInGroup : *this->group_) {
        auto task = dynamic_cast<TaskNodeAbstraction *>(taskInGroup);
        if (task) {
          sum += task->averageExecutionDurationPerElement();
        } else {
          throw std::runtime_error("All the nodes in a group should be of the same type, the representative is of"
                                   " type TaskNodeAbstraction but not the group members.");
        }
      }
      mean = sum / (this->group_->size());

      for (auto taskInGroup : *this->group_) {
        auto task = dynamic_cast<TaskNodeAbstraction *>(taskInGroup);
        if (task) {
          auto diff = (double) (task->averageExecutionDurationPerElement().count() - mean.count());
          sd += diff * diff;
        } else {
          throw std::runtime_error("All the nodes in a group should be of the same type, the representative is of"
                                   " type TaskNodeAbstraction but not the group members.");
        }
      }
      meanSD = {mean, std::chrono::nanoseconds((int64_t) std::sqrt(sd / (double) this->group_->size()))};
    }
    return meanSD;
  }

  /// @brief Accessor to the min / max wait time duration of the nodes in the group
  /// @return Min / max wait time duration in nanoseconds of the nodes in the group
  /// @throw std::runtime_error All the nodes in a group are not of the same types
  [[nodiscard]] std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> minmaxMemoryWaitTimeGroup() const {
    std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> minMax =
        {std::chrono::nanoseconds::zero(), std::chrono::nanoseconds::zero()};
    if (dynamic_cast<TaskNodeAbstraction const *>(this)) {
      auto minMaxElem = std::minmax_element(
          this->group_->cbegin(), this->group_->cend(),
          [](auto lhs, auto rhs) {
            auto lhsAsTask = dynamic_cast<TaskNodeAbstraction const *>(lhs);
            auto rhsAsTask = dynamic_cast<TaskNodeAbstraction const *>(rhs);
            if (lhsAsTask && rhsAsTask) {
              return lhsAsTask->memoryWaitDuration() < rhsAsTask->memoryWaitDuration();
            } else {
              throw std::runtime_error("All the nodes in a group should be of the same type, the representative is of"
                                       " type TaskNodeAbstraction but not the nodes in the group.");
            }
          }
      );
      auto minTask = dynamic_cast<TaskNodeAbstraction *>(*minMaxElem.first);
      auto maxTask = dynamic_cast<TaskNodeAbstraction *>(*minMaxElem.second);
      if (minTask && maxTask) {
        minMax = {minTask->memoryWaitDuration(), maxTask->memoryWaitDuration()};
      } else {
        throw std::runtime_error("All the nodes in a group should be of the same type, the representative is of"
                                 " type TaskNodeAbstraction but not the nodes in the group.");
      }
    }
    return minMax;
  }

  /// @brief Accessor to the mean / standard deviation wait time duration of the nodes in the group
  /// @return Mean / standard deviation wait time duration in nanoseconds of the nodes in the group
  /// @throw std::runtime_error All the nodes in a group are not of the same types
  [[nodiscard]] std::pair<std::chrono::nanoseconds,
                          std::chrono::nanoseconds> meanSDMemoryWaitTimePerElementGroup() const {
    std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> meanSD =
        {std::chrono::nanoseconds::zero(), std::chrono::nanoseconds::zero()};
    if (dynamic_cast<TaskNodeAbstraction const *>(this)) {
      std::chrono::nanoseconds
          sum = std::chrono::nanoseconds::zero(),
          mean = std::chrono::nanoseconds::zero();
      double
          sd = 0;

      for (auto taskInGroup : *this->group_) {
        auto task = dynamic_cast<TaskNodeAbstraction *>(taskInGroup);
        if (task) {
          sum += task->memoryWaitDuration();
        } else {
          throw std::runtime_error("All the nodes in a group should be of the same type, the representative is of"
                                   " type TaskNodeAbstraction but not the group members.");
        }
      }
      mean = sum / (this->group_->size());

      for (auto taskInGroup : *this->group_) {
        auto task = dynamic_cast<TaskNodeAbstraction *>(taskInGroup);
        if (task) {
          auto diff = (double) (task->memoryWaitDuration().count() - mean.count());
          sd += diff * diff;
        } else {
          throw std::runtime_error("All the nodes in a group should be of the same type, the representative is of"
                                   " type TaskNodeAbstraction but not the group members.");
        }
      }
      meanSD = {mean, std::chrono::nanoseconds((int64_t) std::sqrt(sd / (double) this->group_->size()))};
    }
    return meanSD;
  }

  /// @brief Accessor to the min / max number of elements received in the group
  /// @return Min / max number of elements received in the group
  /// @throw std::runtime_error All the nodes in a group are not of the same types
  [[nodiscard]] std::pair<size_t, size_t> minmaxNumberElementsReceivedGroup() const {
    std::pair<size_t, size_t> minMax = {0, 0};

    if (dynamic_cast<TaskNodeAbstraction const *>(this)) {
      auto minMaxElem = std::minmax_element(
          this->group_->cbegin(), this->group_->cend(),
          [](auto lhs, auto rhs) {
            auto lhsAsTask = dynamic_cast<TaskNodeAbstraction const *>(lhs);
            auto rhsAsTask = dynamic_cast<TaskNodeAbstraction const *>(rhs);
            if (lhsAsTask && rhsAsTask) {
              return lhsAsTask->numberReceivedElements() < rhsAsTask->numberReceivedElements();
            } else {
              throw std::runtime_error("All the nodes in a group should be of the same type, the representative is of"
                                       " type TaskNodeAbstraction but not the nodes in the group.");
            }
          }
      );
      auto minTask = dynamic_cast<TaskNodeAbstraction *>(*minMaxElem.first);
      auto maxTask = dynamic_cast<TaskNodeAbstraction *>(*minMaxElem.second);
      if (minTask && maxTask) {
        minMax = {minTask->numberReceivedElements(), maxTask->numberReceivedElements()};
      } else {
        throw std::runtime_error("All the nodes in a group should be of the same type, the representative is of"
                                 " type TaskNodeAbstraction but not the nodes in the group.");
      }
    }

    return minMax;
  }

  /// @brief Accessor to the mean / standard deviation number of elements received in the group
  /// @return Mean / standard deviation number of elements received in the group
  /// @throw std::runtime_error All the nodes in a group are not of the same types
  [[nodiscard]] std::pair<double, double> meanSDNumberElementsReceivedGroup() const {
    std::pair<double, double> meanSD = {0, 0};
    if (dynamic_cast<TaskNodeAbstraction const *>(this)) {
      size_t sum = 0;
      double mean = 0, sd = 0;

      for (auto taskInGroup : *this->group_) {
        auto task = dynamic_cast<TaskNodeAbstraction *>(taskInGroup);
        if (task) {
          sum += task->numberReceivedElements();
        } else {
          throw std::runtime_error("All the nodes in a group should be of the same type, the representative is of"
                                   " type TaskNodeAbstraction but not the nodes in the group.");
        }
      }
      mean = (double) sum / (double) (this->group_->size());

      for (auto taskInGroup : *this->group_) {
        auto task = dynamic_cast<TaskNodeAbstraction *>(taskInGroup);
        if (task) {
          auto diff = (double) task->numberReceivedElements() - mean;
          sd += diff * diff;
        } else {
          throw std::runtime_error("All the nodes in a group should be of the same type, the representative is of"
                                   " type TaskNodeAbstraction but not the nodes in the group.");
        }
      }

      meanSD = {mean, std::sqrt(sd / (double) this->group_->size())};
    }
    return meanSD;
  }

  /// @brief Accessor to the min / max number of elements per input type received in the group
  /// @return Min / max number of elements per input type received in the group
  [[nodiscard]] std::map<std::string, std::pair<size_t, size_t>> minmaxNumberElementsReceivedGroupPerInput() const {
    auto firstIterator = this->group_->cbegin();
    std::map<std::string, std::pair<size_t, size_t>> minMax;

    for (auto const &[t, nbElem] : dynamic_cast<TaskNodeAbstraction const *>(*firstIterator)->nbElementsPerInput()) {
      minMax.insert({t, {nbElem, nbElem}});
    }

    std::advance(firstIterator, 1);

    for (auto it = firstIterator; it != this->group_->cend(); ++it) {
      for (auto const &[t, nbElem] : dynamic_cast<TaskNodeAbstraction const *>(*firstIterator)->nbElementsPerInput()) {
        auto &mm = minMax.at(t);
        mm = {std::min(mm.first, nbElem), std::max(mm.second, nbElem)};
      }
    }
    return minMax;
  }

  /// @brief Accessor to the mean / standard deviation number of elements received per input in the group
  /// @return Mean / standard deviation number of elements received per input in the group
  [[nodiscard]] std::map<std::string, std::pair<double, double>> meanSDNumberElementsReceivedGroupPerInput() const {
    std::map<std::string, std::vector<size_t>> nbElementsGathered;
    std::map<std::string, std::pair<double, double>> ret;

    auto firstIterator = this->group_->cbegin();

    for (auto const &[t, nbElem] : dynamic_cast<TaskNodeAbstraction const *>(*firstIterator)->nbElementsPerInput()) {
      nbElementsGathered.insert({t, {nbElem}});
    }

    std::advance(firstIterator, 1);

    for (auto it = firstIterator; it != this->group_->cend(); ++it) {
      for (auto const &[t, nbElem] : dynamic_cast<TaskNodeAbstraction const *>(*firstIterator)->nbElementsPerInput()) {
        nbElementsGathered.at(t).push_back(nbElem);
      }
    }

    for (auto const &[t, values] : nbElementsGathered) {
      double mean = std::accumulate(values.cbegin(), values.cend(), 0, std::plus<>()), sd = 0;
      mean /= (double) (values.size());
      for (auto const &value : values) {
        auto diff = (double) value - mean;
        sd += diff * diff;
      }
      ret.insert({t, {mean, std::sqrt(sd / (double) values.size())}});
    }

    return ret;
  }

  /// @brief Accessor to the min / max dequeue + execution duration per input of the nodes in the group
  /// @return Min / max dequeue + execution duration per input in nanoseconds of the nodes in the group
  [[nodiscard]] std::map<std::string, std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds>>
  minmaxDequeueExecutionDurationGroupPerInput() const {
    auto firstIterator = this->group_->cbegin();
    std::map<std::string, std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds>> minMax;

    for (auto const &[t, duration]
        : dynamic_cast<TaskNodeAbstraction const *>(*firstIterator)->dequeueExecutionDurationPerInput()) {
      minMax.insert({t, {duration, duration}});
    }

    std::advance(firstIterator, 1);

    for (auto it = firstIterator; it != this->group_->cend(); ++it) {
      for (auto const &[t, duration]
          : dynamic_cast<TaskNodeAbstraction const *>(*firstIterator)->dequeueExecutionDurationPerInput()) {
        auto &mm = minMax.at(t);
        mm = {std::min(mm.first, duration), std::max(mm.second, duration)};
      }
    }
    return minMax;
  }

  /// @brief Accessor to the mean / standard deviation dequeue + execution duration per input of the nodes in the group
  /// @return Mean / standard deviation dequeue + execution duration per input in nanoseconds of the nodes in the group
  [[nodiscard]] std::map<std::string, std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds>>
  meanSDDequeueExecutionDurationGroupPerInput() const {
    std::map<std::string, std::vector<std::chrono::nanoseconds>> nbElementsGathered;
    std::map<std::string, std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds>> ret;

    auto firstIterator = this->group_->cbegin();

    for (auto const &
          [t, nbElem] : dynamic_cast<TaskNodeAbstraction const *>(*firstIterator)->dequeueExecutionDurationPerInput()) {
      nbElementsGathered.insert({t, {nbElem}});
    }

    std::advance(firstIterator, 1);

    for (auto it = firstIterator; it != this->group_->cend(); ++it) {
      for (auto const &
            [t, nbElem] : dynamic_cast<TaskNodeAbstraction const *>(*firstIterator)->dequeueExecutionDurationPerInput()) {
        nbElementsGathered.at(t).push_back(nbElem);
      }
    }

    for (auto const &[t, values] : nbElementsGathered) {
      std::chrono::nanoseconds mean = std::chrono::nanoseconds::zero();
      double sd = 0;
      for (auto const &v : values) { mean += v; }
      mean = mean / values.size();
      for (auto const &value : values) {
        auto diff = (double) (value.count()) - (double) (mean.count());
        sd += diff * diff;
      }
      ret.insert({t, {mean, std::chrono::nanoseconds((int64_t) std::sqrt(sd / (double) this->group_->size()))}});
    }

    return ret;
  }

  /// @brief Accessor to the min / max execution per elements duration per input of the nodes in the group
  /// @return Min / max execution per elements duration per input in nanoseconds of the nodes in the group
  [[nodiscard]] std::map<std::string, std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds>>
  minmaxExecTimePerElementGroupPerInput() const {
    auto firstIterator = this->group_->cbegin();
    std::map<std::string, std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds>> minMax;

    for (auto const &
          [t, duration] : dynamic_cast<TaskNodeAbstraction const *>(*firstIterator)->executionDurationPerInput()) {
      minMax.insert({t, {duration, duration}});
    }

    std::advance(firstIterator, 1);

    for (auto it = firstIterator; it != this->group_->cend(); ++it) {
      for (auto const &
            [t, duration] : dynamic_cast<TaskNodeAbstraction const *>(*firstIterator)->executionDurationPerInput()) {
        auto &mm = minMax.at(t);
        mm = {std::min(mm.first, duration), std::max(mm.second, duration)};
      }
    }
    return minMax;
  }

  /// @brief Accessor to the mean / standard deviation execution per elements duration per input of the nodes in the group
  /// @return >ean / standard deviation execution per elements duration per input in nanoseconds of the nodes in the group
  [[nodiscard]] std::map<std::string, std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds>>
  meanSDExecTimePerElementGroupPerInput() const {
    std::map<std::string, std::vector<std::chrono::nanoseconds>> nbElementsGathered;
    std::map<std::string, std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds>> ret;

    auto firstIterator = this->group_->cbegin();

    for (auto const &
          [t, nbElem] : dynamic_cast<TaskNodeAbstraction const *>(*firstIterator)->executionDurationPerInput()) {
      nbElementsGathered.insert({t, {nbElem}});
    }

    std::advance(firstIterator, 1);

    for (auto it = firstIterator; it != this->group_->cend(); ++it) {
      for (auto const &
            [t, nbElem] : dynamic_cast<TaskNodeAbstraction const *>(*firstIterator)->executionDurationPerInput()) {
        nbElementsGathered.at(t).push_back(nbElem);
      }
    }

    for (auto const &[t, values] : nbElementsGathered) {
      std::chrono::nanoseconds mean = std::chrono::nanoseconds::zero();
      double sd = 0;
      for (auto const &v : values) { mean += v; }
      mean = mean / values.size();
      for (auto const &value : values) {
        auto diff = (double) (value.count()) - (double) (mean.count());
        sd += diff * diff;
      }
      ret.insert({t, {mean, std::chrono::nanoseconds((int64_t) std::sqrt(sd / (double) this->group_->size()))}});
    }

    return ret;
  }

  /// @brief Create a group to insert in the map
  virtual void createGroup(std::map<NodeAbstraction *, std::vector<NodeAbstraction *>> &) = 0;
};
}
}
}
#endif //HEDGEHOG_ANY_COPYABLE_ABSTRACTION_H
