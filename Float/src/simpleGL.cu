////////////////////////////////////////////////////////////////////////////
//
//
//
////////////////////////////////////////////////////////////////////////////

/*
    This example demonstrates how to use the Cuda OpenGL bindings to
    dynamically modify a vertex buffer using a Cuda kernel.

    The steps are:
    1. Create an empty vertex buffer object (VBO)
    2. Register the VBO with Cuda
    3. Map the VBO for writing from Cuda
    4. Run Cuda kernel to modify the vertex positions
    5. Unmap the VBO
    6. Render the results using OpenGL

    Host code
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <array>
#include <iostream>
// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>

#include "kernel.cu"


#include <boost/multi_array.hpp>
#include <cassert>
#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

using namespace std;
using namespace boost;
////////////////////// struct
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

std::array<std::vector<long>, 16384> neighbor_index;

typedef struct Index {
    long id[NUM_NEIGHBOR];
}Index;

//typedef struct neighbor

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width  = 512;
const unsigned int window_height = 512;

const unsigned int mesh_width    = 128;
const unsigned int mesh_height   = 128;
const unsigned int mesh_length   = 128;

int MAXX=128;
int MAXY=128;

int CELLSIZEX=1.0;
int CELLSIZEY=1.0;

#define INVALID_ID 2111111111
/*          2,147,483,648
#define DT     0.09f     // Delta T for interative solver
#define VIS    0.0025f   // Viscosity constant  //do nhot
#define FORCE (5.8f*DIM) // Force scale factor
#define FR     4         // Force update radius
*/

float g_fAnim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling

float avgFPS = 0.0f;
unsigned int frameCount = 0;


int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

GLuint float_vbo;
struct cudaGraphicsResource *float_vbo_cuda_resource;
void *d_float_vbo_buffer = NULL;

////////////////// bien toan cuc luu tru thong tin

FloatType *AllFloats = NULL;

CellType *AllCells_host = NULL;
CellType *AllCells_device;

Index *cell_index_host = NULL;
Index *cell_index_device;

void initCell2D(int CAMode){

	long tempid = 0;
	int num_inactive = 0;
	    for(int j = 0; j < MAXY; ++j){
	      for(int i = 0; i < MAXX; ++i)
	        {
	    	  Position temp;
	    	  temp.x = (float)i/MAXX ;
	    	  temp.y = (float)j/MAXY ;

	    	  AllCells_host[i+MAXY*j].id = tempid;
	    	  AllCells_host[i+MAXY*j].CellPos = temp;
	    	  int state  = rand() % 100 ;
	//    	  cout << " state = " <<state;
	    	  if (state %4 ==0) { //Diep random init
	    		  AllCells_host[i+MAXY*j].state = NORMAL ;
	    	//	  cout << " \n NORMAL id = " <<tempid;
	    	  }else {
	    		  AllCells_host[i+MAXY*j].state = INACTIVE ;
	    	//	  cout << " \n INACTIVE id = " <<tempid;
	    		  num_inactive ++;
	    	  }
	    	  tempid++;
	    	  //Flat[x + HEIGHT* (y + WIDTH* z)]
//	    	  The algorithm is mostly the same. If you have a 3D array Original[HEIGHT, WIDTH, DEPTH] then you could turn it into Flat[HEIGHT * WIDTH * DEPTH] by
//	    	  Flat[x + WIDTH * (y + DEPTH * z)] = Original[x, y, z]
	        }
	    }
	   // cout << " tempid = " <<tempid;
	    if(CAMode==CA_VON_NEUMANN){ //4 neighbor 2D
	    	vector<long> neighbor ;
	    	for(int j = 0; j < MAXY; ++j){
			   for(int i = 0; i < MAXX; ++i)
				{
				   long tempindex[NUM_NEIGHBOR];

				   if (i>0){//left(x) = (x - 1) % M
					   neighbor.push_back(AllCells_host[((i+j*MAXX)-1)].id);
					   tempindex[0] = AllCells_host[((i+j*MAXX)-1)].id ;
				   }else {
					   neighbor.push_back(INVALID_ID);
					   tempindex[0] = INVALID_ID ;
				   }
				   if (i<MAXX-1){//right(x) = (x + 1) % M
					   neighbor.push_back(AllCells_host[((i+j*MAXX)+1)].id);
					   tempindex[1] = AllCells_host[((i+j*MAXX)+1)].id ;
				   }else{
					   neighbor.push_back(INVALID_ID);
					   tempindex[1] = INVALID_ID ;
				   }
				   if (j>0){//above(x) = (x - M) % (M * N)
					   neighbor.push_back(AllCells_host[((i+j*MAXX)-MAXX)%(MAXX*MAXY)].id);
					   tempindex[2] = AllCells_host[((i+j*MAXX)-MAXX)%(MAXX*MAXY)].id ;
				   }else {
					   neighbor.push_back(INVALID_ID);
					   tempindex[2] = INVALID_ID ;
				   }
				   if (j<MAXY-1){//below(x) = (x + M) % (M * N)
					   neighbor.push_back(AllCells_host[((i+j*MAXX)+MAXX)%(MAXX*MAXY)].id);
					   tempindex[3] = AllCells_host[((i+j*MAXX)+MAXX)%(MAXX*MAXY)].id ;
				   }else {
					   neighbor.push_back(INVALID_ID);
					   tempindex[3] = INVALID_ID ;
				   }
				   memcpy(cell_index_host[i+(j*MAXX)].id, tempindex, NUM_NEIGHBOR * sizeof(long)); //CA Diep change size
				//   cell_index_host[i+(j*MAXX)].id = tempindex;
				   neighbor_index[i+(j*MAXX)] = neighbor;
				   neighbor.clear();

		/*		   if(i==2&&j==0){
					   cout << "\n i+j*MAXX= " << i+j*MAXX << " AllCells id= " <<AllCells[(i+j*MAXX)].id << " neightbors: "
							   << AllCells[((i+j*MAXX)-1)%MAXX].id <<","<< AllCells[((i+j*MAXX)+1)%MAXX].id <<","
							   << AllCells[((i+j*MAXX)-MAXX)%(MAXX*MAXY)].id <<","<< AllCells[((i+j*MAXX)+MAXX)%(MAXX*MAXY)].id;

				   }*/
				}
	    	}
	    }
//	printf("\n done initCell maxid = %d , inactive=%d ",tempid,num_inactive);
}

__global__ void game_of_life_kernel(float4 *pos, unsigned int mesh_width,unsigned int mesh_length, int CAMode, CellType *Cells_device)
{
	// __syncthreads();

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

 //   printf("\n in Kernel x,y [%d][%d] position: %2f  - %2f   id: %ld  state:%d ",x,y,Cells_device[y*mesh_width+x].CellPos.x,Cells_device[y*mesh_width+x].CellPos.y,Cells_device[y*mesh_width+x].id,Cells_device[y*mesh_width+x].state);
    if(Cells_device[y*mesh_width+x].state == INACTIVE ){
    	pos[y*mesh_width+x] = make_float4(0,0,0,1.0f);
 //   	Cells_device[cell_index_device[y*mesh_width+x].id[0]].state ++;
 //   	Cells_device[cell_index_device[y*mesh_width+x].id[0]].state %=2;
    //	Cells_device[y*mesh_width+x].state == INACTIVE
    	return ;
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

/*
__global__ void stepCell(FloatState *nowState_d, FloatState *nextState_d, canaux *channels_d, int node_number)
{

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < node_number)
	{
	    nextState_d[idx] = computeCell(nowState_d, idx, channels_d, devStates);
	}
}
*/

/*


__device__ void computeCell(FloatState *nowState_d, int nodeIndex, canaux *channels_d, curandState* devState_d)
{

}
*/


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);
void computeFPS();
// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource,int modeCA,CellType *cells_d);


///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
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


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{

    pArgc = &argc;
    pArgv = argv;

    setenv ("DISPLAY", ":0", 0);

	sdkCreateTimer(&timer);


	int arraycellsize = MAXX*MAXY*sizeof(CellType);
	int arrayindex = MAXX*MAXY*sizeof(Index);
		//Allocating memory of host variables
			AllCells_host = (CellType*) malloc(arraycellsize);
			cell_index_host = (Index*) malloc(arrayindex);
		//	AllFloats = (FloatType *)
		//Allocating memory to device variable

	//initVariable();
	initCell2D(CA_VON_NEUMANN);


	//cout<<" id = 551 [x,y]= [" << AllCells[551].CellPos.x<<","<<AllCells[551].CellPos.y<< "]";
    //cout<< "\n neighbor: ";


	if (false == initGL(&argc, argv))
	{
		return false;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "device"))
	{
		if (gpuGLDeviceInit(argc, (const char **)argv) == -1)
		{
			return false;
		}
	}
	else
	{
		cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
	}


	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutCloseFunc(cleanup);


	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
	createVBO(&float_vbo, &float_vbo_cuda_resource, cudaGraphicsMapFlagsWriteDiscard);
	// run the cuda part
	//runCuda(&cuda_vbo_resource,0);
	//runCuda(&float_vbo_cuda_resource,1);

	glutMainLoop();

}



////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource,int modeCA,CellType *Cells_device)
{
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;

    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
                                                         *vbo_resource));
    //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

    // execute the kernel game_of_life_kernel
	dim3 block(8, 8, 1);
	dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	game_of_life_kernel<<< grid, block>>>(dptr, mesh_width,mesh_length, modeCA,Cells_device);

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
  //*diep*  unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
    unsigned int size = mesh_width * mesh_height  * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}



////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);
    int arraycellsize = MAXX*MAXY*sizeof(CellType);
    checkCudaErrors(cudaMalloc((CellType**)&AllCells_device,arraycellsize));
    checkCudaErrors(cudaMemcpy(AllCells_device, AllCells_host, arraycellsize, cudaMemcpyHostToDevice));
/*
    int arrayindex = MAXX*MAXY*sizeof(Index);
    checkCudaErrors(cudaMalloc(( Index** ) &cell_index_device,arrayindex));
	checkCudaErrors(cudaMemcpy(cell_index_device, cell_index_host, arrayindex, cudaMemcpyHostToDevice));

*/
    // run CUDA kernel to generate vertex positions
    runCuda(&cuda_vbo_resource,0,AllCells_device);
 //   runCuda(&float_vbo_cuda_resource,1,AllCells_device);
  //  cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(AllCells_host,AllCells_device, arraycellsize, cudaMemcpyDeviceToHost));

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glPointSize(2.0f);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);
/*
    //draw float
    glPointSize(2.0f);
	glBindBuffer(GL_ARRAY_BUFFER, float_vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(0.0, 1.0, 0.0);
	glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
	glDisableClientState(GL_VERTEX_ARRAY);
*/
    glutSwapBuffers();

    g_fAnim += 0.01f;

    sdkStopTimer(&timer);
    computeFPS();
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

    // unregister this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    if (vbo)
    {
        deleteVBO(&vbo, cuda_vbo_resource);
    }

    if (float_vbo)
	{
		deleteVBO(&float_vbo, float_vbo_cuda_resource);
	}


    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    /*
	cudaFree(channels_d);
	cudaFree(nowState_d);
	cudaFree(nextState_d);
	cudaFree(buffState_d);
	cudaFree(devState_d);

	free(AllCells_host);
	free(AllFloats);
	free(neighbor_index);
	//free(nextState_h);

	 */
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :

                glutDestroyWindow(glutGetWindow());
                return;

    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "Cuda GL Float: %3.1f fps (Max 100Hz)", avgFPS);
    glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Float");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // initialize necessary OpenGL extensions
    glewInit();

    if (! glewIsSupported("GL_VERSION_2_0 "))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

 //   float attenuation[] = {1.0f, -0.01f, -.000001f};
  //  glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, attenuation, 0);
 //   glPointParameter(GL_POINT_DISTANCE_ATTENUATION,1.0f,-0.01f,-.000001f);
  //  glEnable(GL_POINT_DISTANCE_ATTENTUATION);
    SDK_CHECK_ERROR_GL();

    return true;
}
