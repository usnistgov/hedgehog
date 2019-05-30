//
// Created by anb22 on 5/28/19.
//

#ifndef HEDGEHOG_ABSTRACT_RELEASE_RULE_H
#define HEDGEHOG_ABSTRACT_RELEASE_RULE_H

template<class Data>
class AbstractReleaseRule {
 public:
  virtual void used() = 0;
  virtual bool canRelease() = 0;

};

#endif //HEDGEHOG_ABSTRACT_RELEASE_RULE_H
