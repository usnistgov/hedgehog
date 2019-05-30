//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-03-20.
//

#ifndef HEDGEHOG_HELPER_H
#define HEDGEHOG_HELPER_H

#include <tuple>

//template<class Output, class ...Inputs>
//class AbstractTask;
////template<class Output, class ...Inputs>
////class Rule;
template<class ...Inputs>
class MultiReceivers;
template<class Output>
class Sender;
////template<class ...Inputs>
////class Bookkeeper;

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

//  template<class T>
//  struct HelperMultiConsumerType;

//  template<class...I>
//  struct HelperMultiConsumerType<std::tuple<I...> > {
//    using type = MultiReceivers<I...>;
//  };

//  template<class T>
//  struct HelperBookkeeperType;
//
//  template<class...I>
//  struct HelperBookkeeperType<std::tuple<I...> > {
//    using type = Bookkeeper<I...>;
//  };

//  template<class O, class I>
//  struct HelperRuleType;
//
//  template<class O, class...I>
//  struct HelperRuleType<O, std::tuple<I...> > {
//    using type = Rule<O, I...>;
//  };

//  template<class O, class ...I>
//  struct HelperTaskType;
//
//  template<class O, class ...I>
//  struct HelperTaskType<O, std::tuple<I...>> {
//    using type = AbstractTask<O, I...>;
//  };
//
//  template<class O, class ...I>
//  struct HelperGraphType;
//
//  template<class O, class ...I>
//  struct HelperGraphType<O, std::tuple<I...>> {
//    using type = Graph<O, I...>;
//  };

};

#endif //HEDGEHOG_HELPER_H
