//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-03-20.
//

#ifndef HEDGEHOG_HELPER_H
#define HEDGEHOG_HELPER_H

#include <tuple>

#include "../core/io/base/receiver/core_multi_receivers.h"

template<class ...Inputs>
	class MultiReceivers;

template<class Output>
	class Sender;

struct Helper {
  template<class Inputs>
  struct HelperMultiReceiversType;

  template<class ...Inputs>
  struct HelperMultiReceiversType<std::tuple<Inputs...>> {
    using type = MultiReceivers<Inputs...>;
  };

  template<class Inputs>
  struct HelperCoreMultiReceiversType;

  template<class ...Inputs>
  struct HelperCoreMultiReceiversType<std::tuple<Inputs...>> {
    using type = CoreMultiReceivers<Inputs...>;
  };
};

#endif //HEDGEHOG_HELPER_H
