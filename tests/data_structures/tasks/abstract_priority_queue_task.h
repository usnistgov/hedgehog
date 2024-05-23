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

#ifndef HEDGEHOG_ABSTRACT_PRIORITY_QUEUE_TASK_H
#define HEDGEHOG_ABSTRACT_PRIORITY_QUEUE_TASK_H

#include "../implementors/multi_priority_queue_receivers.h"

template<size_t Separator, class ...AllTypes>
class AbstractPriorityQueueTask : public hh::AbstractTask<Separator, AllTypes...> {
 public:
  explicit AbstractPriorityQueueTask(std::string const &name, size_t const numberThreads, bool const automaticStart)
      : hh::AbstractTask<Separator, AllTypes...>(
      std::make_shared<hh::core::CoreTask<Separator, AllTypes...>>(
  this,
  name, numberThreads, automaticStart,
  std::make_shared<hh::core::implementor::DefaultSlot>(),
      std::make_shared<MPQR<Separator, AllTypes...>>(),
  std::make_shared<hh::tool::DME<Separator, AllTypes...>>(this),
  std::make_shared<hh::core::implementor::DefaultNotifier>(),
      std::make_shared<hh::tool::MDS<Separator, AllTypes...>>()
  )
  ) {}
};

#endif //HEDGEHOG_ABSTRACT_PRIORITY_QUEUE_TASK_H
