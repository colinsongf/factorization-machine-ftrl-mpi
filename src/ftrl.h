#ifndef FTRL_H
#define FTRL_H
#include "load_data.h"
#include "mpi.h"
#include <math.h>

class FTRL{
    public:
        FTRL(Load_Data* load_data, int total_num_proc, int my_rank)
                    : data(load_data), num_proc(total_num_proc), rank(my_rank){
                                init();
        }
        ~FTRL(){}
        void init(){
            v_dim = data->glo_fea_dim*factor;
            temp_value = new float[data->glo_fea_dim]();
            glo_w = new float[data->glo_fea_dim]();
            loc_w = new float[data->glo_fea_dim]();
            loc_v=new float[v_dim]();
            loc_v_arr=new float*[data->glo_fea_dim];
            for (int i = 0; i < data->glo_fea_dim; i++)
                    loc_v_arr[i]=&loc_v[i*factor];

            glo_v = new float[v_dim]();
            glo_v_arr = new float*[data->glo_fea_dim];
            for(int i = 0; i < data->glo_fea_dim; i++){
                    glo_v_arr[i] = &glo_v[i*factor];
            }
            loc_g_w = new float[data->glo_fea_dim]();
            glo_g_w = new float[data->glo_fea_dim]();
            loc_g_v=new float[v_dim]();
            loc_g_v_arr=new float*[data->glo_fea_dim];
            for (int i = 0; i < data->glo_fea_dim; i++)
                    loc_g_v_arr[i]=&loc_g_v[i*factor];
            glo_g_v = new float[v_dim]();
            glo_g_v_arr = new float*[data->glo_fea_dim];
            for(int i = 0; i < data->glo_fea_dim; i++){
                    glo_g_v_arr[i] = &glo_g_v[i*factor];
            }
            loc_z_w = new float[data->glo_fea_dim]();
            loc_sigma_w = new float[data->glo_fea_dim]();
            loc_n_w = new float[data->glo_fea_dim]();
            loc_z_v=new float[v_dim]();
            loc_z_v_arr=new float*[data->glo_fea_dim];
            for (int i = 0; i < data->glo_fea_dim; i++)
                    loc_z_v_arr[i]=&loc_z_v[i*factor];
            loc_sigma_v = new float[v_dim]();
            loc_sigma_v_arr = new float*[data->glo_fea_dim];
            for (int i = 0; i < data->glo_fea_dim; i++)
                    loc_sigma_v_arr[i]=&loc_sigma_v[i*factor];
            loc_n_v=new float[v_dim]();
            loc_n_v_arr=new float*[data->glo_fea_dim];
            for (int i = 0; i < data->glo_fea_dim; i++)
                    loc_n_v_arr[i]=&loc_n_v[i*factor];
            alpha = 1.0;
            beta = 1.0;
            lambda1 = 0.0;
            lambda2 = 1.0;
            steps = 5;
            batch_size = 0;
            bias = 2.0;
        }
        float sigmoid(float x){
            if(x < -30) return 1e-6;
            else if(x > 30) return 1.0;
            else{
                    double ex = pow(2.718281828, x);
                    return ex / (1.0 + ex);
            }
        }

        void update_other_parameter(){
            MPI_Status status;
            if(rank != 0){
                    MPI_Send(loc_g_w, data->glo_fea_dim, MPI_FLOAT, 0, 99, MPI_COMM_WORLD);
                    MPI_Send(loc_g_v, data->glo_fea_dim*factor, MPI_FLOAT, 0, 999, MPI_COMM_WORLD);
            }
            else if(rank == 0){
                    for(int f_idx = 0; f_idx < data->glo_fea_dim; f_idx++){
                            glo_g_w[f_idx] = loc_g_w[f_idx];
                    }
                    for(int ranknum = 1; ranknum < num_proc; ranknum++){
                            MPI_Recv(loc_g_w, data->glo_fea_dim, MPI_FLOAT, ranknum, 99, MPI_COMM_WORLD, &status);
                            for(int j = 0; j < data->glo_fea_dim; j++){
                                    glo_g_w[j] += loc_g_w[j];
                            }

                            for(int j = 0; j < data->glo_fea_dim; j++){
                                    for(int k = 0; k < factor; k++){
                                            glo_g_v_arr[j][k] = loc_g_v_arr[j][k];
                                    }
                            }
                            MPI_Recv(loc_g_v, data->glo_fea_dim*factor, MPI_FLOAT, ranknum, 999, MPI_COMM_WORLD, &status);
                            for(int j = 0; j < data->glo_fea_dim; j++){
                                    for(int k = 0; k < factor; k++){
                                            glo_g_v_arr[j][k] = loc_g_v_arr[j][k];
                                    }
                            }
                    }
                    for(int col = 0; col < data->glo_fea_dim; col++){
                            loc_sigma_w[col] = (sqrt(loc_n_w[col] + glo_g_w[col] * glo_g_w[col]) - sqrt(loc_n_w[col])) / alpha;
                            loc_z_w[col] += glo_g_w[col] - loc_sigma_w[col] * loc_w[col];
                            loc_n_w[col] += glo_g_w[col] * glo_g_w[col];

                            for(int k = 0; k < factor; k++){
                                    loc_sigma_v_arr[col][k] = (sqrt(loc_n_v_arr[col][k] + glo_g_v_arr[col][k] * glo_g_v_arr[col][k]) - sqrt(loc_n_v_arr[col][k])) / alpha;
                                    loc_z_v_arr[col][k] += glo_g_v_arr[col][k] - loc_sigma_v_arr[col][k] * loc_v_arr[col][k];
                                    loc_n_v_arr[col][k] += glo_g_v_arr[col][k] * glo_g_v_arr[col][k];
                            }
                    }
            }
        }

        void update_v(){
            for(long int j = 0; j < data->glo_fea_dim; j++){
                    for(int k = 0; k < factor; k++){
                            float tmp_z = loc_z_v_arr[j][k];
                            continue;
                            if(abs(tmp_z) <= lambda1){
                                    loc_v_arr[j][k] = 0.0;
                            }
                            else{
                                    float tmpr = 0.0;
                                    if(tmp_z >= 0){
                                            tmpr = tmp_z - lambda1;
                                    }
                                    else{
                                            tmpr = tmp_z + lambda1;
                                    }
                                    float tmpl = -1 * ( ( beta + sqrt(loc_n_v_arr[j][k]) ) / alpha  + lambda2);
                                    loc_v_arr[j][k] = tmpr / tmpl;
                            }
                    }
            }
        }

        void update_w(){
            for(int col = 0; col < data->glo_fea_dim; col++){
                    if(abs(loc_z_w[col]) <= lambda1){
                            loc_w[col] = 0.0;
                    }
                    else{
                            float tmpr= 0.0;
                            if(loc_z_w[col] >= 0) tmpr = loc_z_w[col] - lambda1;
                            else tmpr = loc_z_w[col] + lambda1;
                            float tmpl = -1 * ( ( beta + sqrt(loc_n_w[col]) ) / alpha  + lambda2);
                            loc_w[col] = tmpr / tmpl;
                    }
            }
        }

        void update_parameter(){
            MPI_Status status;
            if(rank == 0){
                    update_w();
                    update_v();
                    for(int r = 1; r < num_proc; r++){
                            MPI_Send(loc_w, data->glo_fea_dim, MPI_FLOAT, r, 99, MPI_COMM_WORLD);
                            MPI_Send(loc_v, data->glo_fea_dim*factor, MPI_FLOAT, r, 999, MPI_COMM_WORLD);
                    }
            }
            else if(rank != 0){
                    MPI_Recv(glo_w, data->glo_fea_dim, MPI_FLOAT, 0, 99, MPI_COMM_WORLD, &status);
                    for(int j = 0; j < data->glo_fea_dim; j++){
                            loc_w[j] = glo_w[j];
                    }
                    MPI_Recv(glo_v, data->glo_fea_dim*factor, MPI_FLOAT, 0, 999, MPI_COMM_WORLD, &status);
                    for(int j = 0; j < data->glo_fea_dim*factor; j++){
                            loc_v[j] = glo_v[j];
                    }
            }
        }

        void ftrl(){
            MPI_Status status;
            long int index = 0; int value = 1; float pctr = 0.0;
            for(int i = 0; i < steps; i++){
                    int row = i * batch_size;
                    loss_sum = 0.0;
                    update_parameter();
                    std::cout<<"step "<<i<<std::endl;
                    while( (row < (i + 1) * batch_size) && (row < data->fea_matrix.size()) ){
                            float wx = bias;
                            for(int col = 0; col < data->fea_matrix[row].size(); col++){//for one instance
                                    index = data->fea_matrix[row][col].idx;
                                    value = data->fea_matrix[row][col].val;
                                    wx += loc_w[index] * value;
                            }
                            for(int k = 0; k < factor; k++){
                                    float vxvx = 0.0, vvxx = 0.0;
                                    for(int col = 0; col < data->fea_matrix[row].size(); col++){
                                            index = data->fea_matrix[row][col].idx;
                                            value = data->fea_matrix[row][col].val;
                                            vxvx += loc_v_arr[col][k] * value;
                                            vvxx += loc_v_arr[col][k] * loc_v_arr[col][k] * value*value;
                                    }
                                    vxvx *= vxvx;
                                    vxvx -= vvxx;
                                    wx += vxvx * 1.0 / 2;
                            }
                            pctr = sigmoid(wx);
                            loss_sum = (pctr - data->label[row]);
                            for(int l = 0; l < data->glo_fea_dim; l++){
                                    loc_g_w[l] += loss_sum * value;
                                    float vx = 0;
                                    for(int k = 0; k < factor; k++){
                                            for(int j = 0; j != l && j < data->glo_fea_dim; j++){
                                                    if(loc_v_arr[j][k] == 0.0) continue;
                                                    vx +=  loc_v_arr[j][k] * value;
                                            }
                                            loc_g_v_arr[l][k] += loss_sum * vx;
                                    }
                            }
                            ++row;
                    }
                    update_other_parameter();
            }//end for
        }

public:
	int batch_size;
  	int steps;
	long int v_dim;
	float* glo_w;
	float* loc_w;
	float* glo_v;
	float** glo_v_arr;
	float* loc_v;
	float** loc_v_arr;
private:
	Load_Data* data;
	float* temp_value; 
    float* loc_f_val;
	float* loc_g_w;
	float* glo_g_w;
	float* loc_g_v;
	float** loc_g_v_arr;
	float* glo_g_v;
	float** glo_g_v_arr;

	float* loc_z_w;
	float* loc_sigma_w;
	float* loc_n_w;
 	float* loc_z_v;
	float** loc_z_v_arr;
	float* loc_sigma_v;
	float** loc_sigma_v_arr;
	float* loc_n_v;
	float** loc_n_v_arr;

	int factor;
	float alpha;
	float beta;
	float lambda1;
	float lambda2;
	float bias;
	
	float loss_sum;
	int num_proc;
	int rank;
};

#endif
