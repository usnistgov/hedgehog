//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_THREADABLE_H
#define HEDGEHOG_THREADABLE_H

class Threadable {
 private:
  size_t numberCopies_ = 0;
 public:
  explicit Threadable(size_t const &numberCopies) : numberCopies_(numberCopies) {}
  virtual ~Threadable() = default;

 public:
  size_t numberCopies() const { return numberCopies_; }
  void numberCopies(size_t numberCopies) { numberCopies_ = numberCopies; }
};

#endif //HEDGEHOG_THREADABLE_H
