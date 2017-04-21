#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>
#include<curand.h>
#include<curand_kernel.h>
int maxx=256;
int maxy=256;
int maxz=128;
int cellsizex=1.0;
int cellsizey=1.0;
int cellsizez=1.0;


typedef struct
{
	int x;
	int y;
	int z;
}Direction;

struct CellPosition {
    int x;
    int y;
    int z;
};

typedef struct{
	//float temperature;
	//presure
	//salinity
	//trajectory
	int floatState;//NORMAL |
}FloatState;

/*
typedef struct{
	CellPosition cellPosition;
	NodeState nodeState;
	TreeState tree;
}Node;
*/

#define NORMAL 1
#define DRIFT 2

//extern void initFloatState(FloatState *node_state, int node_number, int num_starting_float);

//extern __device__ TreeState computeStateForest(TreeState *nowState_d, int nodeIndex, canaux *channels_d, curandState* devState_d);

//extern __global__ void stepStateForest(TreeState *nowState_d, TreeState *nextState_d, canaux *channels_d, int node_number, curandState *devStates_d);

/*
extern __global__ void setup_kernel(curandState *state, unsigned long seed);

__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
}


//------------------------------------------------------------------------------------
__device__ float generateNumber(curandState* globalState, int nodeIndex)
{
    curandState localState = globalState[nodeIndex];
    float random = curand_uniform( &localState );
    globalState[nodeIndex] = localState;
    return random;
}
*/
//Randomly firing some places
void initFloatState(FloatState *node_state, int node_number,int num_starting_float)
{
	int node;

	int startingNode[num_starting_float];

	for (int i = 0; i < node_number; i++)
	{
	    node_state[i].floatState = NORMAL;
	}

	for (int i = 0; i < num_starting_float; i++)
	{
	    startingNode[i] = -1;
	}

	srand(time(NULL));

	for (int i = 0; i < num_starting_float; i++)
	{

	    while(1)
	    {
			bool fired = false;
				node = rand() % node_number;
			for (int j = 0; j < i; j++)
			{
				if (startingNode[j] == node)
				{
				fired = true;
				break;
				}
			}
			if (fired == false)
			{
			   startingNode[i] = node;
			   break;
			}
	    }
	    node_state[node].floatState = DRIFT;
	}
}

//-------------------------------------------------------------------------------------------
/**   Version 1.0
***** One cell fired if one of its neighbor is fired. If it fired, it changes to ash. If it is ash, it will become empty.
*****
*/
/*
__device__ FloatState computeStateForest(FloatState *nowState_d, int nodeIndex, canaux *channels_d, curandState* devState_d)
{
	FloatState myState;

	myState = nowState_d[nodeIndex];

	//Checking its neighbours
	int nbIn = channels_d[nodeIndex].nbIn;

	if (myState.treeState == NORMAL)
	{
	   int nodeIn;
	   for (int i = 0; i < nbIn; i++)
	   {
     	    	nodeIn = channels_d[nodeIndex].read[i].node;

	    	if (nowState_d[nodeIn].treeState == FIRED)
	    	{
		    myState.treeState = FIRED;
		    break;
	    	}
	   }
	}
	else if (myState.treeState == FIRED)
	{
	   myState.treeState = ASH;
	}else if (myState.treeState == ASH)
	{
	   myState.treeState = EMPTY;
	}
	return myState;
}

/**
*This function the changing the state of each cell of the  grid
*
*/

/*
__global__ void stepStateForest(FloatState *nowState_d, FloatState *nextState_d, canaux *channels_d, int node_number, curandState *devStates)
{

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < node_number)
	{
	    nextState_d[idx] = computeStateForest(nowState_d, idx, channels_d, devStates);
	}
}
*/

