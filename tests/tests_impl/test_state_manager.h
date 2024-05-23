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


#include "../data_structures/tasks/abc_to_def_task.h"
#include "../data_structures/states/def_to_abc_state.h"


void testSharedState() {
  size_t nbResults = 0, nbResultsD = 0, nbResultsE = 0, nbResultsF = 0;
  auto inputTask = std::make_shared<ABCToDEFTask>("Input");
  auto intermediateTask = std::make_shared<ABCToDEFTask>("Intermediate");
  auto outputTask = std::make_shared<ABCToDEFTask>("Output");
  auto state = std::make_shared<DEFToABCState>();
  auto sm = std::make_shared<hh::StateManager<3, D, E, F, A, B, C>>(state, "My SM");
  auto sm2 = std::make_shared<hh::StateManager<3, D, E, F, A, B, C>>(state, "My SM2");

  hh::Graph<3, A, B, C, D, E, F> g("My Graph");

  g.inputs(inputTask);
  g.edges(inputTask, sm);
  g.edges(sm, intermediateTask);
  g.edges(intermediateTask, sm2);
  g.edges(sm2, outputTask);
  g.outputs(outputTask);

  g.executeGraph();

  for(int i = 0; i < 10; ++i){
    g.pushData(std::make_shared<A>());
    g.pushData(std::make_shared<B>());
    g.pushData(std::make_shared<C>());
  }
  g.finishPushingData();

  while(auto res = g.getBlockingResult()){
    ++nbResults;
    std::visit(hh::ResultVisitor{
        [&nbResultsD]([[maybe_unused]]std::shared_ptr<D> &val) { ++nbResultsD; },
        [&nbResultsE]([[maybe_unused]]std::shared_ptr<E> &val) { ++nbResultsE; },
        [&nbResultsF]([[maybe_unused]]std::shared_ptr<F> &val) { ++nbResultsF; }
    }, *res);
  }
  ASSERT_EQ(nbResults, 7290);
  ASSERT_EQ(nbResultsD, 2430);
  ASSERT_EQ(nbResultsE, 2430);
  ASSERT_EQ(nbResultsF, 2430);

  g.waitForTermination();
}

