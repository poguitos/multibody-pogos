// mbd_crosscheck: writes this library's answers for the comparison with
// independent codes (plan task 5.3; see README.md).
//
//   mbd_crosscheck trees <output.json> [count=40] [seed=1]        for compare_pinocchio.py
//   mbd_crosscheck mechanisms <output.json> [count=12] [seed=1]   for compare_chrono.py

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "export.hpp"

int main(int argc, char** argv)
{
    const bool trees = argc >= 3 && std::strcmp(argv[1], "trees") == 0;
    const bool mechanisms = argc >= 3 && std::strcmp(argv[1], "mechanisms") == 0;
    if (!trees && !mechanisms) {
        std::cerr << "usage: mbd_crosscheck trees <output.json> [count=40] [seed=1]\n"
                     "       mbd_crosscheck mechanisms <output.json> [count=12] [seed=1]\n";
        return 2;
    }
    const int count = argc > 3 ? std::atoi(argv[3]) : (trees ? 40 : 12);
    const unsigned long long seed = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 1;
    return trees ? mbd_crosscheck::export_trees(argv[2], count, seed)
                 : mbd_crosscheck::export_mechanisms(argv[2], count, seed);
}
