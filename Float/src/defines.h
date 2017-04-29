
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
	float x;
	float y;
	float z;
}Direction;

struct Position {
    float x;
    float y;
    float z;
};

//float state
#define INACTIVE 0
#define NORMAL 1
#define DRIFT 2

#define MAX_MEASURE_SIZE 6
#define MAX_TRAJECTORY_SIZE 6

//cell
#define CA_VON_NEUMANN 1
#define CA_MOORE 2
#define NUM_NEIGHBOR 4

#define INVALID_ID 2111111111

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width  = 512;
const unsigned int window_height = 512;

/*
const unsigned int mesh_width    = 128;
const unsigned int mesh_height   = 128;
const unsigned int mesh_length   = 128;
*/
typedef struct {
	float pressure;
	float salinity;
	float temperature;
}FloatMeasurement;

typedef struct {
	//date date
	Position FloatPos; //presure
	FloatMeasurement *measure;
	int measure_size;
}FloatTrajectoryPoint;

typedef struct{
	int id;
	int floatState;//NORMAL |
	FloatTrajectoryPoint* trajectory;
	int trajectory_size;
}FloatType;

typedef struct celltype{
	float temperature;
	float depth; //presure
	float salinity;
	//velocity
	//force
	Position CellPos;//position
	int state; //INACTIVE | NORMAL | DRIFT
	long id;
}CellType;

typedef struct Index {
    long id[NUM_NEIGHBOR];
}Index;


