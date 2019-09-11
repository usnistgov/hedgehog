////
//// Created by anb22 on 6/20/19.
////

#include "hedgehog/tools/traits.h"
#include "hedgehog/tools/helper.h"

#include <iostream>
#include <iomanip>

template<class ...T>
class Data {
 public:
  using inputs_t = std::tuple<T...>;
};

template<
    class D1,
    class D2,
    class D1_t = typename D1::inputs_t,
    class D2_t = typename D2::inputs_t,
    class isInputCompatible = std::enable_if_t<HedgehogTraits::is_included_v<D1_t, D2_t>>>
void testTemplates() {
  std::cout << std::setw(160) << __PRETTY_FUNCTION__ << ": \t" << HedgehogTraits::is_included_v<D1_t, D2_t>
            << std::endl;

}

int main(){
//  testTemplates<Data<size_t>, Data<int>>();
//  testTemplates<Data<size_t>, Data<int, float>>();
  testTemplates<Data<size_t, float>, Data<int, size_t >>();
  testTemplates<Data<>, Data<size_t>>();
}
