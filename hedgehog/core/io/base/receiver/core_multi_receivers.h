//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_CORE_MULTI_RECEIVERS_H
#define HEDGEHOG_CORE_MULTI_RECEIVERS_H

#include "core_receiver.h"
#include "core_slot.h"
#include "../../../node/core_node.h"

template<class ...Inputs>
class CoreMultiReceivers :
    public virtual CoreSlot, public virtual CoreReceiver<Inputs> ... {
 public:
  CoreMultiReceivers(std::string_view const &name, NodeType const type, size_t const numberThreads) : CoreSlot(name,
                                                                                                               type,
                                                                                                               numberThreads),
                                                                                                      CoreReceiver<
                                                                                                          Inputs>(name,
                                                                                                                  type,
                                                                                                                  numberThreads)... {
    HLOG_SELF(0, "Creating CoreMultiReceivers with type: " << (int) type << " and name: " << name)
  }

  ~CoreMultiReceivers() override {
    HLOG_SELF(0, "Destructing CoreMultiReceivers")
  }

  virtual bool receiversEmpty() = 0;

};

#endif //HEDGEHOG_CORE_MULTI_RECEIVERS_H
