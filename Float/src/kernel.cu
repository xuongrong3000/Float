#include "defines.h"


//void initFloatState(FloatState *node_state, int node_number, int num_starting_float);

//extern __device__ TreeState computeStateForest(TreeState *nowState_d, int nodeIndex, canaux *channels_d, curandState* devState_d);

__global__ void runfloat(float4 *pos, unsigned int maxx,unsigned int maxy, unsigned maxz, int CAMode, CellType *Cells_device,Index * index_device)
{

}

__device__ void stepCell(unsigned long long int idx, int CAMode, CellType *Cells_device,Index * index_device,bool showMode){
//	pos[y*mesh_width+x] = make_float4(0,0,0,1.0f); ||(!showMode&&i<4)
	for (int i=0; i<NUM_NEIGHBOR;i++){
		if(index_device[idx].id[i]!=INVALID_ID)
		{
		/*	if(idx ==10){
				printf("\n before index idx=10: %d", Cells_device[index_device[idx].id[i]].state);
			}
		*/
			Cells_device[index_device[idx].id[i]].state ++;
			Cells_device[index_device[idx].id[i]].state %=2;
		}
	}
}

__global__ void game_of_life_kernel(float4 *pos, unsigned int maxx,unsigned int maxy, unsigned int maxz, int CAMode, CellType *Cells_device,Index * index_device,bool showMode)
{
	// __syncthreads();
/*	const unsigned long long int blockId = blockIdx.x //1D
	        + blockIdx.y * gridDim.x //2D
	        + gridDim.x * gridDim.y * blockIdx.z; //3D

	// global unique thread index, block dimension uses only x-coordinate
	const unsigned long long int threadId = blockId * blockDim.x + threadIdx.x;
*/
	unsigned long long int threadId ;
	if (showMode){ //3D
		threadId 	= (blockIdx.x + blockIdx.y * gridDim.x 	+ gridDim.x * gridDim.y * blockIdx.z ) * (blockDim.x * blockDim.y * blockDim.z)
											+ (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
		if(Cells_device[threadId].state == INACTIVE ){
	     	pos[threadId] = make_float4(0,0,0,1.0f);
	     	stepCell(threadId,CAMode,Cells_device,index_device,showMode);
		}else{
		 pos[threadId] = make_float4(Cells_device[threadId].CellPos.x, Cells_device[threadId].CellPos.z, Cells_device[threadId].CellPos.y, 1.0f);
		 Cells_device[threadId].state = INACTIVE;
		}
	}else { //2D
		unsigned long long int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned long long int y = blockIdx.y*blockDim.y + threadIdx.y;
		threadId = y*maxx + x;

		if(Cells_device[threadId].state == INACTIVE ){
			pos[threadId] = make_float4(0,0,0,1.0f);
			stepCell(threadId,CAMode,Cells_device,index_device,showMode);
		}else{
			pos[threadId] = make_float4(Cells_device[threadId].CellPos.x, 0.5f, Cells_device[threadId].CellPos.y, 1.0f);
			Cells_device[threadId].state = INACTIVE;
		}
	}
   /* unsigned long long int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned long long int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned long long int z = blockIdx.z*blockDim.z + threadIdx.z;
 */
//    printf("\n in Kernel z,y,x=[%d][%d][%d] index0: %ld  cellID: %ld  state:%d",z,y,x,index_device[x+maxz*(y+maxy*z)].id[0],Cells_device[x+maxz*(y+maxy*z)].id,Cells_device[x+maxz*(y+maxy*z)].state);
 //   printf("\n in Kernel y,x [%d][%d] position: %2f  - %2f   id: %ld  state:%d ",y,x,Cells_device[y*mesh_width+x].CellPos.x,Cells_device[y*mesh_width+x].CellPos.y,Cells_device[y*mesh_width+x].id,Cells_device[y*mesh_width+x].state);
    //Flat[x + HEIGHT* (y + WIDTH* z)]

 //   	pos[threadId] = make_float4(Cells_device[threadId].CellPos.x, Cells_device[threadId].CellPos.z, Cells_device[threadId].CellPos.y, 1.0f);
 //   	Cells_device[index_device[y*mesh_width+x].id[0]].state = NORMAL;
 //   	Cells_device[cell_index_device[y*mesh_width+x].id[0]].state %=2;
    //	Cells_device[y*mesh_width+x].state == INACTIVE


	// get neighbor
  //  if (x% 5 == 0|| y%3 ==0) //not visible - skip it
  //  	return ;

    // write output vertex
 // pos[y*width+x] = make_float4(u, w, v, 1.0f);  (x,z,y,alpha);
  //  if( Cells_d[y*mesh_width+x].state==NORMAL|| Cells_d[y*mesh_width+x].state==DRIFT){

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

