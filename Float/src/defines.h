#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector>
#include <array>





using namespace std;


typedef struct
{
	int x;
	int y;
}Direction;

struct Position {
    float x;
    float y;
};

//float state
#define INACTIVE 0
#define NORMAL 1
#define DRIFT 2

#define CA_VON_NEUMANN 1
#define CA_MOORE 2
#define NUM_NEIGHBOR 4

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width  = 512;
const unsigned int window_height = 512;

const unsigned int mesh_width    = 128;
const unsigned int mesh_height   = 128;
const unsigned int mesh_length   = 128;



#define INVALID_ID 2111111111

typedef struct {
	float pressure;
	float salinity;
	float temperature;
}FloatMeasurement;

typedef struct {
	//date date
	Position FloatPos; //presure
	vector<FloatMeasurement> measure;
	int measure_size;
}FloatTrajectoryPoint;

typedef struct{
	int id;
	int floatState;//NORMAL |
	vector<FloatTrajectoryPoint> trajectory;
	int trajectory_size;
}FloatType;

typedef struct celltype{
	float temperature;
	float height; //presure
	float salinity;
	//velocity
	//force
	Position CellPos;//position
	int state; //INACTIVE | NORMAL | DRIFT
	long id;
}CellType;

//std::array<std::vector<long>, 16384> neighbor_index;

typedef struct Index {
    long id[NUM_NEIGHBOR];
}Index;

//typedef struct neighbor


