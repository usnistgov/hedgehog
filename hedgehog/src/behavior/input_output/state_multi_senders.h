

#ifndef HEDGEHOG_STATE_MULTI_SENDERS_H_
#define HEDGEHOG_STATE_MULTI_SENDERS_H_

#include "state_sender.h"
#include "multi_senders.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog behavior namespace
namespace behavior {

/// @brief Behavior abstraction for states that send multiple types of data
/// @tparam Outputs Types of data the state sends
template<class ...Outputs>
class StateMultiSenders : public MultiSenders<Outputs...>, public StateSender<Outputs>... {
 public:
  /// @brief Default constructor
  StateMultiSenders() = default;

  /// @brief Default destructor
  ~StateMultiSenders()  override = default;

  /// @brief Add result to the ready list
  /// @tparam DataType Type of the data, should be part of the state Output types
  /// @param data Data of type DataType added to the ready list
  template<tool::MatchOutputTypeConcept<Outputs...> DataType>
  void addResult(std::shared_ptr<DataType> data) { StateSender<DataType>::readyList()->push(data); }

};
}
}
#endif //HEDGEHOG_STATE_MULTI_SENDERS_H_
