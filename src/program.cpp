#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
using namespace std;

#include <filesystem> //require C++17
namespace fs = std::filesystem;

// #include "ann/dataset/DSFactory.h"
// #include "ann/loader/annheader.h"
// #include "config/Config.h"
#include "../demo/hash/xMapDemo.h"
// #include "../demo/hash/HeapDemo.h"
#include "list/listheader.h"

// #include "loader/dataloader.h"
// #include "loader/dataset.h"
// #include "modelzoo/threeclasses.h"
// #include "modelzoo/twoclasses.h"
// #include "optim/Adagrad.h"
// #include "optim/Adam.h"
// #include "sformat/fmt_lib.h"
// #include "tensor/xtensor_lib.h"

int main(int argc, char **argv) {
    // dataloader:
    // case_data_wo_label_1();
    // case_data_wi_label_1();
    // case_batch_larger_nsamples();

    // Classification:
    // twoclasses_classification();
    // threeclasses_classification();
    cout << "Demo 1" << endl;
    // hashDemo1();
    hashDemo1();
    cout << endl;
    cout << "Demo 2" << endl;
    // hashDemo2();
    hashDemo2();
    cout << endl;
    cout << "Demo 3" << endl;
    // hashDemo3();
    hashDemo3();
    cout << endl;
    cout << "Demo 4" << endl;
    hashDemo4();
    cout << endl;
    cout << "Demo 5" << endl;
    hashDemo5();
    cout << endl;
    cout << "Demo 6" << endl;
    hashDemo6();
    cout << endl;
    cout << "Demo 7" << endl;
    hashDemo7();
    cout << endl;
    // cout << hash30();
    return 0;
}
