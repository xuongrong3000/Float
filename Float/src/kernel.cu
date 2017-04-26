#include "defines.h"


//void initFloatState(FloatState *node_state, int node_number, int num_starting_float);

//extern __device__ TreeState computeStateForest(TreeState *nowState_d, int nodeIndex, canaux *channels_d, curandState* devState_d);

__global__ void runfloat(float4 *pos, unsigned int mesh_width,unsigned int mesh_length, int CAMode, CellType *Cells_device,Index * index_device)
{

}

__device__ void stepCell(unsigned int idx, unsigned int mesh_width,unsigned int mesh_length, int CAMode, CellType *Cells_device,Index * index_device){
//	pos[y*mesh_width+x] = make_float4(0,0,0,1.0f);
	    	if(index_device[idx].id[0]!=INVALID_ID){
	    		Cells_device[index_device[idx].id[0]].state ++;
				Cells_device[index_device[idx].id[0]].state %=2;
	    	}
	    	if(index_device[idx].id[1]!=INVALID_ID){
				Cells_device[index_device[idx].id[1]].state ++;
				Cells_device[index_device[idx].id[1]].state %=2;
			}
	    	if(index_device[idx].id[2]!=INVALID_ID){
				Cells_device[index_device[idx].id[2]].state ++;
				Cells_device[index_device[idx].id[2]].state %=2;
			}
	    	if(index_device[idx].id[3]!=INVALID_ID){
				Cells_device[index_device[idx].id[3]].state ++;
				Cells_device[index_device[idx].id[3]].state %=2;
			}

	 //   	Cells_device[cell_index_device[y*mesh_width+x].id[0]].state %=2;
	 //	Cells_device[y*mesh_width+x].state == INACTIVE
	    	return ;

}

__global__ void game_of_life_kernel(float4 *pos, unsigned int mesh_width,unsigned int mesh_length, int CAMode, CellType *Cells_device,Index * index_device)
{
	// __syncthreads();

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
//    printf("\n in Kernel y,x [%d][%d] index: %ld  cellID: %ld ",y,x,index_device[y*mesh_width+x].id[0],Cells_device[y*mesh_width+x].id);
 //   printf("\n in Kernel y,x [%d][%d] position: %2f  - %2f   id: %ld  state:%d ",y,x,Cells_device[y*mesh_width+x].CellPos.x,Cells_device[y*mesh_width+x].CellPos.y,Cells_device[y*mesh_width+x].id,Cells_device[y*mesh_width+x].state);
    if(Cells_device[y*mesh_width+x].state == INACTIVE ){
    	pos[y*mesh_width+x] = make_float4(0,0,0,1.0f);
 //   	Cells_device[index_device[y*mesh_width+x].id[0]].state = NORMAL;
 //   	Cells_device[cell_index_device[y*mesh_width+x].id[0]].state %=2;
    //	Cells_device[y*mesh_width+x].state == INACTIVE
    	stepCell(y*mesh_width+x,mesh_width,mesh_length,CAMode,Cells_device,index_device);
    }else
	// get neighbor
  //  if (x% 5 == 0|| y%3 ==0) //not visible - skip it
  //  	return ;

    // write output vertex
 // pos[y*width+x] = make_float4(u, w, v, 1.0f);  (x,z,y,alpha);
  //  if( Cells_d[y*mesh_width+x].state==NORMAL|| Cells_d[y*mesh_width+x].state==DRIFT){
		if(CAMode ==0){ //vonneuman
			  pos[y*mesh_width+x] = make_float4(Cells_device[y*mesh_width+x].CellPos.x, 0.5f, Cells_device[y*mesh_width+x].CellPos.y, 1.0f);
			  Cells_device[y*mesh_width+x].state == INACTIVE;
		}else {
			  pos[y*mesh_width+x] = make_float4(Cells_device[y*mesh_width+x].CellPos.x, 1.0f, Cells_device[y*mesh_width+x].CellPos.y, 1.0f);
			  Cells_device[y*mesh_width+x].state == INACTIVE;
		}
   // }
}

__global__ void simple_conveyor_kernel(float4 *pos, unsigned int mesh_width,unsigned int mesh_length, int CAMode)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	//z =
    // calculate uv coordinates
    float u = x / (float) mesh_width;
    float v = y / (float) mesh_length;
    u = u*2.0f - 1.0f;
    v = v*2.0f - 1.0f;

    if (x% 5 == 0|| y%3 ==0)
    	return ;
    // calculate simple sine wave pattern
  //  float freq = 4.0f;
  //  float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

    // write output vertex
 // pos[y*width+x] = make_float4(u, w, v, 1.0f);  (x,z,y,alpha);

    if(CAMode ==0){ //vonneuman
    	  pos[y*mesh_width+x] = make_float4(u, 0.5f, v, 1.0f);

	}else {
		  pos[y*mesh_width+x] = make_float4(u, 1.0f, v, 1.0f);
	}
}
/*
extern __global__ void setup_kernel(curandState *state, unsigned long seed);

__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
}

*/
//Randomly firing some places

