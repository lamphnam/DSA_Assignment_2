#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
using namespace std;

#include <filesystem> //require C++17
namespace fs = std::filesystem;

#include "ann/annheader.h"
#include "config/Config.h"
#include "dataset/DSFactory.h"
#include "list/listheader.h"
#include "loader/dataloader.h"
#include "loader/dataset.h"
#include "modelzoo/threeclasses.h"
#include "modelzoo/twoclasses.h"
#include "optim/Adagrad.h"
#include "optim/Adam.h"
#include "sformat/fmt_lib.h"
#include "tensor/xtensor_lib.h"

int main(int argc, char **argv) {
    // dataloader:
    // case_data_wo_label_1();
    // case_data_wi_label_1();
    // case_batch_larger_nsamples();

    // Classification:
    twoclasses_classification();
    // threeclasses_classification();

    return 0;
}
