#ifndef LOAD_DATA_H_
#define LOAD_DATA_H_

#include <fstream>
#include <iostream>
#include <vector>
#include <set>
#include "mpi.h"

#define MASTER_ID (0)
#define FEA_DIM_FLAG (99)

struct sparse_feature{
    long int idx;
    int val;
};

class Load_Data {
public:
    std::ifstream fin_;
    std::vector<std::vector<sparse_feature> > fea_matrix;
    std::vector<sparse_feature> key_val;
    sparse_feature sf;
    std::vector<int> label;
    std::string line;
    std::set<long int> feaIdx;
    std::set<long int>::iterator setIter;
    int y, value, nchar;
    long int index;
    long int loc_fea_dim = 0;
    long int glo_fea_dim = 0;

    Load_Data(const char *file_name){
        fin_.open(file_name, std::ios::in);
        if(!fin_.is_open()){
            std::cout<<"open file error: "<<file_name << std::endl;
            exit(1);
        } 
        std::cout<<file_name<<std::endl;
    }
    ~Load_Data(){
        fin_.close();
    }

    void load_data_batch(int nproc, int rank){
        MPI_Status status;
        fea_matrix.clear();
        //std::cout<<"load batch data start..."<<std::endl;
        while(!fin_.eof()){
            std::getline(fin_, line);
            if(fin_.eof()) break;
            key_val.clear();
            const char *pline = line.c_str();
            if(sscanf(pline, "%d%n", &y, &nchar) >= 1){
                pline += nchar;
                label.push_back(y);
                while(sscanf(pline, "%ld:%d%n", &index, &value, &nchar) >= 2){
                    pline += nchar;
                    sf.idx = index;
                    if(index+1 > loc_fea_dim) loc_fea_dim = index+1;
                    setIter = feaIdx.find(index);
                    if(setIter == feaIdx.end()){
                        feaIdx.insert(index);
                    }
                    sf.val = value;
                    key_val.push_back(sf);
                }
            }
            fea_matrix.push_back(key_val);
        }
        if(rank != 0) {
            MPI_Send(&loc_fea_dim, 1, MPI_LONG, 0, 90, MPI_COMM_WORLD);
        }
        else if(rank == 0){ 
            for(int i = 1; i < nproc; i++){
                MPI_Recv(&loc_fea_dim, 1, MPI_LONG, i, 90, MPI_COMM_WORLD, &status);
                if(loc_fea_dim > glo_fea_dim) glo_fea_dim = loc_fea_dim;
            }
        }
        MPI_Bcast(&glo_fea_dim, 1, MPI_LONG, 0, MPI_COMM_WORLD);//must be in all processes code;
    }
    long int loc_ins_num = 0;
private:
};
#endif
