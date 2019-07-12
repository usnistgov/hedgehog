#include "ep/if_to_iep.h"
#include "tasks/epif_to_i.h"
#include "tasks/if_to_i.h"
#include "tasks/if_to_i_ep_in.h"

void testEP() {
  for(int r = 0; r < 100; ++r) {
    int count = 0;
    std::vector<int> deviceIds = {0, 0, 0};

    auto
        outterGraph = std::make_shared<Graph<int, int, float>>("Output"),
        epGraph = std::make_shared<Graph<int, int, float>>("epGRaph");

    auto
        intputTask = std::make_shared<IFToI>("InputTask"),
        outputTask = std::make_shared<IFToI>("OutputTask");

    auto
        inputEPTask = std::make_shared<IFToIEpIn>("InputEPTask");

    auto
        outputEPTask = std::make_shared<EPIFToI>("OutputEPTask");

    epGraph->input(inputEPTask);
    epGraph->addEdge(inputEPTask, outputEPTask);
    epGraph->output(outputEPTask);

    auto ep = std::make_shared<IFToIEP>("ExecPipeline", epGraph, 3, deviceIds);

    outterGraph->input(intputTask);
    outterGraph->addEdge(intputTask, ep);
    outterGraph->addEdge(ep, outputTask);
    outterGraph->output(outputTask);

    outterGraph->executeGraph();

    for (int i = 0; i < 100; ++i) { outterGraph->pushData(std::make_shared<int>(i)); }
    outterGraph->finishPushingData();

    while (auto data = outterGraph->getBlockingResult()) { ++count; }
    ASSERT_EQ(count, 300);
    outterGraph->waitForTermination();
  }
}