#include <string>
#include <math.h>
#include "load_data.h"
#include "predict.h"
#include "ftrl.h"
#include "mpi.h"
//#include "gtest/gtest.h"

int main(int argc,char* argv[]){  
    int rank, nproc;
    int namelen = 1024;
    char processor_name[namelen];
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    MPI_Get_processor_name(processor_name,&namelen);
    // Initialize Google's logging library.
    //google::InitGoogleLogging(argv[0]);
    // FLAGS_log_dir = "./log";
    int stepnum = atoi(argv[2]);
    int batchsize = atoi(argv[3]);
    char train_data_path[1024];
    const char *train_data_file = argv[4];
    snprintf(train_data_path, 1024, "%s-%05d", train_data_file, rank);
    char test_data_path[1024];
    const char *test_data_file = argv[5];
    snprintf(test_data_path, 1024, "%s-%05d", test_data_file, rank);

    Load_Data train_data(train_data_path); 
    train_data.load_data_batch(nproc, rank);
    
    std::vector<float> model;
    if(strcmp(argv[1], "ftrl") == 0){
        FTRL ftrl(&train_data, nproc, rank);
        ftrl.steps = stepnum;
        ftrl.batch_size = batchsize;
        ftrl.ftrl();
        for(int j = 0; j < train_data.glo_fea_dim; j++){
	        //std::cout<<"w["<< j << "]: "<<ftrl.loc_w[j]<<std::endl;
	        model.push_back(ftrl.loc_w[j]);
        }
    }
    Load_Data test_data(test_data_path);
    test_data.load_data_batch(nproc, rank);
    Predict predict(&test_data, nproc, rank);
    predict.run(model);
   
    MPI::Finalize();
    return 0;
}
