

#ifndef HEDGEHOG_MULTI_QUEUE_RECEIVERS_H_
#define HEDGEHOG_MULTI_QUEUE_RECEIVERS_H_

#include "queue_receiver.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog implementor namespace
namespace implementor {

/// @brief Concrete implementation of the receiver core abstraction for multiple types using a std::queue
/// @tparam Inputs List of input types
template<class ...Inputs>
class MultiQueueReceivers : public QueueReceiver<Inputs> ... {
 public:
  /// @brief Constructor using an abstraction for callbacks
  explicit MultiQueueReceivers(): QueueReceiver<Inputs>()... {}

  /// Default destructor
  virtual ~MultiQueueReceivers() = default;
};

}
}
}
#endif //HEDGEHOG_MULTI_QUEUE_RECEIVERS_H_
