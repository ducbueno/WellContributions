#include <vector>
#include <fstream>
#include <iostream>
#include "kernel.hpp"
#include "WellContributions.hpp"

WellContributions::~WellContributions(){
    delete[] h_Cnnzs;
    delete[] h_Dnnzs;
    delete[] h_Bnnzs;
    delete[] h_Ccols;
    delete[] h_Bcols;
    delete[] h_val_pointers;
    delete[] h_x;
    delete[] h_y;
    delete[] real_y;
}

template<typename T>
int WellContributions::read_arr(char const *fname, T **pp){
    T value;
    std::ifstream input(fname);
    std::vector<T> temp;

    while(input >> value){
        temp.push_back(value);
    }
    input.close();

    T *p;
    p = (T *)malloc(temp.size()*sizeof(T));
    for(std::size_t i = 0; i < temp.size(); i++){
        p[i] = temp[i];
    }
    *pp = p;

    return temp.size();
}

void WellContributions::read_data(char *fnum){
    std::string base_dir = "../data/real/";
    std::string num(fnum);
    std::string xname = base_dir + "x" + num + ".txt";
    std::string ybname = base_dir + "y_before" + num + ".txt";
    std::string yaname = base_dir + "y_after" + num + ".txt";

    len_Cnnzs = read_arr<double>("../data/real/Cnnzs.txt", &h_Cnnzs);
    len_Dnnzs = read_arr<double>("../data/real/Dnnzs.txt", &h_Dnnzs);
    len_Bnnzs = read_arr<double>("../data/real/Bnnzs.txt", &h_Bnnzs);
    len_Ccols = read_arr<int>("../data/real/Ccols.txt", &h_Ccols);
    len_Bcols = read_arr<int>("../data/real/Bcols.txt", &h_Bcols);
    len_val_pointers = read_arr<int>("../data/real/Cnnzs.txt", &h_val_pointers);
    len_x = read_arr<double>(xname.c_str(), &h_x);
    len_y_before = read_arr<double>(ybname.c_str(), &h_y);
    len_y_after = read_arr<double>(yaname.c_str(), &real_y);
}

void WellContributions::initialize(){
    cl_int err = CL_SUCCESS;

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.size() == 0){
        std::cout << "No OpenCL platforms found. Aborting." << std::endl;
        exit(0);
    }

    std::string platform_info;
    for(unsigned int i = 0; i < platforms.size(); ++i){
        platforms[i].getInfo(CL_PLATFORM_NAME, &platform_info);
        std::cout << "Platform name      : " << platform_info << std::endl;
        platforms[i].getInfo(CL_PLATFORM_VENDOR, &platform_info);
        std::cout << "Platform vendor    : " << platform_info << std::endl;
        platforms[i].getInfo(CL_PLATFORM_VERSION, &platform_info);
        std::cout << "Platform version   : " << platform_info << std::endl;
        platforms[i].getInfo(CL_PLATFORM_PROFILE, &platform_info);
        std::cout << "Platform profile   : " << platform_info << std::endl;
        platforms[i].getInfo(CL_PLATFORM_EXTENSIONS, &platform_info);
        std::cout << "Platform extensions: " << platform_info << std::endl << std::endl;
    }

    std::cout << "Chosen:\n";
    platforms[platformID].getInfo(CL_PLATFORM_NAME, &platform_info);
    std::cout << "Platform name      : " << platform_info << std::endl;
    platforms[platformID].getInfo(CL_PLATFORM_VERSION, &platform_info);
    std::cout << "Platform version   : " << platform_info << std::endl;

    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[platformID])(), 0};
    context.reset(new cl::Context(CL_DEVICE_TYPE_GPU, properties));

    std::vector<cl::Device> devices = context->getInfo<CL_CONTEXT_DEVICES>();
    if (devices.size() == 0){
        std::cout << "No OpenCL devices found. Aborting.";
        exit(0);
    }
    std::cout << "Found " << devices.size() << " OpenCL devices" << std::endl;

    std::string device_info;
    for (unsigned int i = 0; i < devices.size(); ++i) {
        std::vector<size_t> work_sizes;
        std::vector<cl_device_partition_property> partitions;

        devices[i].getInfo(CL_DEVICE_NAME, &device_info);
        std::cout << "CL_DEVICE_NAME            : " << device_info << std::endl;
        devices[i].getInfo(CL_DEVICE_VENDOR, &device_info);
        std::cout << "CL_DEVICE_VENDOR          : " << device_info << std::endl;
        devices[i].getInfo(CL_DRIVER_VERSION, &device_info);
        std::cout << "CL_DRIVER_VERSION         : " << device_info << std::endl;
        devices[i].getInfo(CL_DEVICE_BUILT_IN_KERNELS, &device_info);
        std::cout << "CL_DEVICE_BUILT_IN_KERNELS: " << device_info << std::endl;
        devices[i].getInfo(CL_DEVICE_PROFILE, &device_info);
        std::cout << "CL_DEVICE_PROFILE         : " << device_info << std::endl;
        devices[i].getInfo(CL_DEVICE_OPENCL_C_VERSION, &device_info);
        std::cout << "CL_DEVICE_OPENCL_C_VERSION: " << device_info << std::endl;
        devices[i].getInfo(CL_DEVICE_EXTENSIONS, &device_info);
        std::cout << "CL_DEVICE_EXTENSIONS      : " << device_info << std::endl;

        devices[i].getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &work_sizes);
        for (unsigned int j = 0; j < work_sizes.size(); ++j) {
            std::cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES[" << j << "]: " << work_sizes[j] << std::endl;
        }
        devices[i].getInfo(CL_DEVICE_PARTITION_PROPERTIES, &partitions);
        for (unsigned int j = 0; j < partitions.size(); ++j) {
            std::cout << "CL_DEVICE_PARTITION_PROPERTIES[" << j << "]: " << partitions[j] << std::endl;
        }
        partitions.clear();
        devices[i].getInfo(CL_DEVICE_PARTITION_TYPE, &partitions);
        for (unsigned int j = 0; j < partitions.size(); ++j) {
            std::cout << "CL_DEVICE_PARTITION_PROPERTIES[" << j << "]: " << partitions[j] << std::endl;
        }

        // C-style properties
        cl_device_id tmp_id = devices[i]();
        cl_ulong size;
        clGetDeviceInfo(tmp_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);
        std::cout << "CL_DEVICE_LOCAL_MEM_SIZE       : " << size / 1024 << " KB" << std::endl;
        clGetDeviceInfo(tmp_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);
        std::cout << "CL_DEVICE_GLOBAL_MEM_SIZE      : " << size / 1024 / 1024 / 1024 << " GB" << std::endl;
        clGetDeviceInfo(tmp_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_ulong), &size, 0);
        std::cout << "CL_DEVICE_MAX_COMPUTE_UNITS    : " << size << std::endl;
        clGetDeviceInfo(tmp_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &size, 0);
        std::cout << "CL_DEVICE_MAX_MEM_ALLOC_SIZE   : " << size / 1024 / 1024 << " MB" << std::endl;
        clGetDeviceInfo(tmp_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_ulong), &size, 0);
        std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE  : " << size << std::endl;
        clGetDeviceInfo(tmp_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);
        std::cout << "CL_DEVICE_GLOBAL_MEM_SIZE      : " << size / 1024 / 1024 / 1024 << " GB" << std::endl << std::endl;
    }

    std::cout << "Chosen:" << std::endl;
    devices[deviceID].getInfo(CL_DEVICE_NAME, &device_info);
    std::cout << "CL_DEVICE_NAME            : " << device_info << std::endl;
    devices[deviceID].getInfo(CL_DEVICE_VERSION, &device_info);
    std::cout << "CL_DEVICE_VERSION         : " << device_info << std::endl;

    cl::Program::Sources source(1, std::make_pair(kernel_s, strlen(kernel_s)));
    cl::Program program_ = cl::Program(*context, source);
    program_.build(devices);

    cl::Event event;
    queue.reset(new cl::CommandQueue(*context, devices[deviceID], 0, &err));

    d_Cnnzs = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * len_Cnnzs);
    d_Dnnzs = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * len_Dnnzs);
    d_Bnnzs = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * len_Bnnzs);
    d_Ccols = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * len_Ccols);
    d_Bcols = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * len_Bcols);
    d_val_pointers = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * len_val_pointers);
    d_x = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * len_x);
    d_y = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * len_y_before);

    kernel.reset(new cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&,
                                    cl::Buffer&, cl::Buffer&, cl::Buffer&,
                                    cl::Buffer&, const unsigned int,
                                    const unsigned int, cl::Buffer&,
                                    cl::LocalSpaceArg, cl::LocalSpaceArg,
                                    cl::LocalSpaceArg>(cl::Kernel(program_, "BCRSMatrixProduct")));

}

void WellContributions::apply_kernel(){
    const unsigned int work_group_size = 32;
    const unsigned int total_work_items = (len_val_pointers - 1)*work_group_size;
    const unsigned int dim = 3;
    const unsigned int dim_wells = 4;
    const unsigned int lmem1 = sizeof(double)*work_group_size;
    const unsigned int lmem2 = sizeof(double)*dim_wells;

    cl::Event event = (*kernel)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
                                d_Cnnzs, d_Dnnzs, d_Bnnzs, d_Ccols, d_Bcols, d_x, d_y, dim, dim_wells, d_val_pointers,
                                cl::Local(lmem1), cl::Local(lmem2), cl::Local(lmem2));
}

void WellContributions::print_results(){
    queue->enqueueReadBuffer(d_y, CL_TRUE, 0, sizeof(double) * len_y_after, h_y);

    std::cout << "\th_y\t\treal_y\t\tdiff" << std::endl;
    for(int i = 0; i < len_y_after; i++){
        std::cout << i << "\t" << h_y[i] << "\t\t" << real_y[i] << "\t\t" << real_y[i] - h_y[i] << std::scientific << std::endl;
    }
}
