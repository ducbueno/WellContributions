#define PROGRAM_FILE "/hdd/Drive/Doutorado/Projeto/Codigos/testes/opencl/WellContributions/src/WellContributions.cl"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <CL/cl.h>

cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename){
    cl_program program;
    FILE *program_handle;
    char *program_buffer;
    size_t program_size;
    cl_int err;

    program_handle = fopen(filename, "r");
    if(program_handle == NULL){
        std::cout << "Couldn't find the program file" << std::endl;
        exit(1);
    }

    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    program = clCreateProgramWithSource(ctx, 1, (const char **)&program_buffer, &program_size, &err);
    if(err < 0){
        std::cout << "Couldn't create the program" << std::endl;
        exit(1);
    }

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    
    return program;
}

template<typename T>
int read_vec(char const *fname, T **pp){
    T value;
    std::ifstream input(fname);
    std::vector<T> temp;
    
    while(input >> value){
        temp.push_back(value);
    }
    input.close();

    T *p;
    p = (T *)malloc(temp.size()*sizeof(T));
    for(int i = 0; i < temp.size(); i++){
        p[i] = temp[i];
    }
    *pp = p;

    return temp.size();
}

template<typename T>
void zero_fill(T **pp, int len){
    T *p;

    p = (T *)malloc(len*sizeof(T));
    for(int i = 0; i < len; i++){
        p[i] = 0;
    }
    *pp = p;
}

int main(){
    cl_int d_blnc, d_blnr;
    cl_mem d_valsB, d_valsD, d_valsC, d_rowptr, d_colsB, d_colsC, d_x, d_y;

    cl_platform_id cpPlatform;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    float *h_valsB;
    int len_valsB = read_vec<float>("../data/valsB.txt", &h_valsB);
    float *h_valsD;
    int len_valsD = read_vec<float>("../data/valsD.txt", &h_valsD);
    float *h_valsC;
    int len_valsC = read_vec<float>("../data/valsC.txt", &h_valsC);
    int *h_rowptr;
    int len_rowptr = read_vec<int>("../data/rowptr.txt", &h_rowptr);
    int *h_colsB;
    int len_colsB = read_vec<int>("../data/colsB.txt", &h_colsB);
    int *h_colsC;
    int len_colsC = read_vec<int>("../data/colsC.txt", &h_colsC);
    float *h_x;
    int len_x = read_vec<float>("../data/x.txt", &h_x);
    float *h_y, *real_y;
    int len_y = read_vec<float>("../data/y.txt", &real_y);
    
    zero_fill<float>(&h_y, len_y);

    size_t localSize = 32;
    size_t globalSize = 5*localSize; // 5 work-groups 
    
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    std::cout << "Got platform and device info!" << std::endl;

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if(err < 0){
        std::cerr << "Couldn't create a context" << std::endl;
        exit(1);
    }
    std::cout << "Created context!" << std::endl;

    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if(err < 0){
        std::cout << "Couldn't create a queue" << std::endl;
        exit(1);
    }
    std::cout << "Created command queue!" << std::endl;

    program = build_program(context, device_id, PROGRAM_FILE);
    std::cout << "Built program!" << std::endl;

    size_t len = 0;
    err = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    char *buffer = static_cast<char*>(calloc(len, sizeof(char)));
    err = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
    std::cout << buffer << std::endl;

    kernel = clCreateKernel(program, "BSRMatrixVectorProduct", &err);
    if(err < 0){
        std::cout << "Couldn't create the kernel (error " << err << ")" << std::endl;
        exit(1);
    }
    std::cout << "Created kernel!" << std::endl;

    d_valsB = clCreateBuffer(context, CL_MEM_READ_ONLY, len_valsB*sizeof(float), NULL, NULL);
    d_valsD = clCreateBuffer(context, CL_MEM_READ_ONLY, len_valsD*sizeof(float), NULL, NULL);
    d_valsC = clCreateBuffer(context, CL_MEM_READ_ONLY, len_valsC*sizeof(float), NULL, NULL);
    d_rowptr = clCreateBuffer(context, CL_MEM_READ_ONLY, len_rowptr*sizeof(int), NULL, NULL);
    d_colsB = clCreateBuffer(context, CL_MEM_READ_ONLY, len_colsB*sizeof(int), NULL, NULL);
    d_colsC = clCreateBuffer(context, CL_MEM_READ_ONLY, len_colsC*sizeof(int), NULL, NULL);
    d_x = clCreateBuffer(context, CL_MEM_READ_ONLY, len_x*sizeof(float), NULL, NULL);
    d_y = clCreateBuffer(context, CL_MEM_READ_WRITE, len_y*sizeof(float), NULL, NULL);
    d_blnc = 3; 
    d_blnr = 4;
    std::cout << "Created buffers!" << std::endl;

    err  = clEnqueueWriteBuffer(queue, d_valsB, CL_TRUE, 0, len_valsB*sizeof(float), h_valsB, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_valsD, CL_TRUE, 0, len_valsD*sizeof(float), h_valsD, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_valsC, CL_TRUE, 0, len_valsC*sizeof(float), h_valsC, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_rowptr, CL_TRUE, 0, len_rowptr*sizeof(int), h_rowptr, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_colsB, CL_TRUE, 0, len_colsB*sizeof(int), h_colsB, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_colsC, CL_TRUE, 0, len_colsC*sizeof(int), h_colsC, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_x, CL_TRUE, 0, len_x*sizeof(float), h_x, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_y, CL_TRUE, 0, len_y*sizeof(float), h_y, 0, NULL, NULL);
    if(err < 0){
        std::cout << "Couldn't write to buffers (error: " << err << ")" << std::endl;
        exit(1);
    }
    std::cout << "Wrote to buffers!" << std::endl;

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_valsC);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_valsD);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_valsB);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_colsC);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_colsB);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_x);
    err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_y);
    err |= clSetKernelArg(kernel, 7, sizeof(cl_int), &d_blnc);
    err |= clSetKernelArg(kernel, 8, sizeof(cl_int), &d_blnr);
    err |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &d_rowptr);
    err |= clSetKernelArg(kernel, 10, sizeof(cl_mem), NULL);
    err |= clSetKernelArg(kernel, 11, sizeof(cl_mem), NULL);
    err |= clSetKernelArg(kernel, 12, sizeof(cl_mem), NULL);
    if(err < 0){
        std::cout << "Couldn't set kernel arguments" << std::endl;
        exit(1);
    }
    std::cout << "Kernel arguments set!" << std::endl;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    if(err < 0){
        std::cout << "Couldn't enqueue kernel" << std::endl;
        exit(1);
    }

    clFinish(queue);
    clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0, len_y*sizeof(float), h_y, 0, NULL, NULL);

    std::cout << std::fixed;
    std::cout << std::setprecision(3);
    std::cout << "\th_y\t\treal_y\t\tdiff" << std::endl;
    for(int i = 0; i < len_y; i++){
        std::cout << i << "\t" << h_y[i] << "\t\t" << real_y[i] << "\t\t" << real_y[i] - h_y[i] << std::endl;
    }

    clReleaseMemObject(d_valsB);
    clReleaseMemObject(d_valsD);
    clReleaseMemObject(d_valsC);
    clReleaseMemObject(d_rowptr);
    clReleaseMemObject(d_colsB);
    clReleaseMemObject(d_colsC);
    clReleaseMemObject(d_x);
    clReleaseMemObject(d_y);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(h_valsB);
    free(h_valsD);
    free(h_valsC);
    free(h_rowptr);
    free(h_colsB);
    free(h_colsC);
    free(h_x);
    free(h_y);

    std::cout << "Program finished!!!" << std::endl;

    return 0;
}
