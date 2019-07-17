//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-07-17.
//

#include "ep/iifep_partial_input.h"
#include "task/task_fi_partial_input.h"
#include "task/task_ii_partial_input.h"

void testPartialInputEP(){
  for(int r = 0; r < 100; ++r) {
	size_t duplication = 10;
	std::vector<int> deviceIds(duplication, 0);
	size_t count = 0;
	auto og = std::make_shared<Graph<int, int, float, double>>();
	auto ig = std::make_shared<Graph<int, int, float>>();
	auto t = std::make_shared<TaskIIPartialInput>();
	auto t2 = std::make_shared<TaskIIPartialInput>();
	ig->input(t);
	ig->addEdge(t, t2);
	ig->output(t2);
	auto ep = std::make_shared<IIFEPPartialInput>(ig, duplication, deviceIds);
	auto it = std::make_shared<TaskFIPartialInput>();
	og->input(ep);
	og->input(it);
	auto ot = std::make_shared<TaskIIPartialInput>();
	og->output(ot);
	og->output(it);
	og->addEdge(ep, ot);
	og->addEdge(it, ot);
	og->executeGraph();
	for (int i = 0; i < 100; ++i) {
	  og->pushData(std::make_shared<int>(i));
	  og->pushData(std::make_shared<float>(i));
	  og->pushData(std::make_shared<double>(i));
	}

	og->finishPushingData();
	while (og->getBlockingResult()) { ++count; }
	ASSERT_EQ(count, 1200);
	og->waitForTermination();
  }
}

void testSimplePartialInput(){
  for(int r = 0; r < 100; ++r) {
	size_t count = 0;

	auto g = std::make_shared<Graph<int, int, float>>();
	auto t = std::make_shared<TaskIIPartialInput>();
	auto t2 = std::make_shared<TaskIIPartialInput>();

	g->input(t);
	g->addEdge(t, t2);
	g->output(t2);

	g->executeGraph();

	for (int i = 0; i < 100; ++i) {
	  g->pushData(std::make_shared<int>(i));
	  g->pushData(std::make_shared<float>(i));
	}

	g->finishPushingData();

	while (g->getBlockingResult()) { ++count; }
	ASSERT_EQ(count, 100);
	g->waitForTermination();
  }
}