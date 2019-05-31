//
// Created by anb22 on 5/13/19.
//

#ifndef HEDGEHOG_LOGGER_H
#define HEDGEHOG_LOGGER_H

#ifdef HLOG_ENABLED
#include <glog/logging.h>

#define HLOG(level, msg) if (HLOG_LEVEL < level) {} else LOG(INFO) << msg;
#define HLOG_SELF(level, msg) if (HLOG_LEVEL < level) {} else LOG(INFO) << this->name() << "(" << this->id() << "/" << (int)this->threadId() <<  "): " << msg;
#define HLOG_NODE(level, node, msg) if (HLOG_LEVEL < level) {} else LOG(INFO) << node->name() << "(" << node->id()  << "/" << (int)this->threadId() <<  "): " << msg;

#else
#define HLOG(level, msg) {}
#define HLOG_SELF(level, msg) {}
#define HLOG_NODE(level, node, msg) {}
#endif

#endif //HEDGEHOG_LOGGER_H
