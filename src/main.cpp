#include "WellContributions.hpp"

int main(int argc, char *argv[]) {
    WellContributions wellContributions;

    wellContributions.read_data(argv[1]);
    wellContributions.initialize();
    wellContributions.apply_kernel();
    wellContributions.print_results();
   
    return 0;
}
