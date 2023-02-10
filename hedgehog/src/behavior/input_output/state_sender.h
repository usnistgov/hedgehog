

#ifndef HEDGEHOG_STATE_SENDER_H_
#define HEDGEHOG_STATE_SENDER_H_

#include <queue>
#include <memory>

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog behavior namespace
namespace behavior {

/// @brief Behavior abstraction for states that send a type of data, holds a ready list for that type
/// @tparam Outputs Type of data the state sends
template<class Output>
class StateSender {
 private:
  std::unique_ptr<std::queue<std::shared_ptr<Output>>>
      readyList_ = std::make_unique<std::queue<std::shared_ptr<Output>>>(); ///< Ready list for that type
 public:
  /// @brief Default constructor
  StateSender() = default;
  /// @brief Default destructor
  ~StateSender() = default;

  /// @brief Ready list accessor
  /// @return Ready list containing all the results data produced by the state
  std::unique_ptr<std::queue<std::shared_ptr<Output>>> const &readyList() const { return readyList_; }

};
}
}
#endif //HEDGEHOG_STATE_SENDER_H_
