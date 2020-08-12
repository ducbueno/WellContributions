#ifndef __WELLCONTRIBUTIONS_H_
#define __WELLCONTRIBUTIONS_H_

#include <memory>
#include <CL/cl.hpp>

class WellContributions{
    private:
        int len_Cnnzs, len_Dnnzs, len_Bnnzs;
        int len_Ccols, len_Bcols, len_val_pointers;
        int len_x, len_y_before, len_y_after;

        double *h_Cnnzs = nullptr;
        double *h_Dnnzs = nullptr;
        double *h_Bnnzs = nullptr;
        int *h_Ccols = nullptr;
        int *h_Bcols = nullptr;
        int *h_val_pointers = nullptr;
        double *h_x = nullptr;
        double *h_y = nullptr;
        double *real_y = nullptr;

        int platformID = 0;
        int deviceID = 0;
        std::unique_ptr<cl::Context> context;
        std::unique_ptr<cl::CommandQueue> queue;
        std::unique_ptr<cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&,
                                        cl::Buffer&, cl::Buffer&, cl::Buffer&,
                                        cl::Buffer&, const unsigned int,
                                        const unsigned int, cl::Buffer&,
                                        cl::LocalSpaceArg, cl::LocalSpaceArg,
                                        cl::LocalSpaceArg> > kernel;

        cl::Buffer d_Cnnzs, d_Dnnzs, d_Bnnzs, d_x, d_y;
        cl::Buffer d_Ccols, d_Bcols, d_val_pointers;

        template<typename T> int read_arr(char const *fname, T **p);
        void read_data(char *fnum);
        void initialize();
        void copy_data_to_gpu();
        void apply_kernel();
        void print_results();

    public:
        WellContributions() {};
        ~WellContributions();
        void run(char *fnum);
};

#endif // __WELLCONTRIBUTIONS_H_
