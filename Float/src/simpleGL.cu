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
	int z;
}Direction;

struct Position {
    int x;
    int y;
    int z;
};

//float state
#define NORMAL 1
#define DRIFT 2
#define CA_VON_NEUMANN 0
#define CA_MOORE 1

typedef struct {
	float pressure;
	float salinity;
	float temperature;
}FloatMeasurement;

typedef struct {
	//date date
	Position FloatPos; //presure
	vector<FloatMeasurement> measure;
}FloatTrajectoryPoint;

typedef struct{
	int id;
	int floatState;//NORMAL |
	vector<FloatTrajectoryPoint> trajectory;
}FloatType;

typedef struct{
	float temperature;
	float height; //presure
	float salinity;
	//velocity
	//force
	Position CellPos;//position
}CellType;

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width  = 512;
const unsigned int window_height = 512;

const unsigned int mesh_width    = 256;
const unsigned int mesh_height   = 256;
const unsigned int mesh_length   = 256;

int MAXX=256;
int MAXY=256;
int MAXZ=128;

int CELLSIZEX=1.0;
int CELLSIZEY=1.0;
int CELLSIZEZ=1.0;

/*
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
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

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
CellType *AllCells = NULL;
FloatType *AllFloats = NULL;

typedef multi_array<CellType, 3> array3DCellType;
typedef array3DCellType::index a3D_index;
array3DCellType Cells(extents[MAXX][MAXY][MAXZ]);

vector<CellType> getNeighbors(int CAmode,CellType aCell)
{
	vector<CellType> result;
	if(CAmode == CA_VON_NEUMANN){ //6 rules
		if(aCell.CellPos.x>0){
			array<a3D_index,3> idx = {{aCell.CellPos.x-1,aCell.CellPos.y,aCell.CellPos.z}};
			result.push_back(Cells(idx)) ;
		}
		if(aCell.CellPos.x<MAXX-1){
			array<a3D_index,3> idx = {{aCell.CellPos.x+1,aCell.CellPos.y,aCell.CellPos.z}};
			result.push_back(Cells(idx)) ;
		}
		if(aCell.CellPos.y>0){
			array<a3D_index,3> idx = {{aCell.CellPos.x,aCell.CellPos.y-1,aCell.CellPos.z}};
			result.push_back(Cells(idx)) ;
		}
		if(aCell.CellPos.y<MAXY-1){
			array<a3D_index,3> idx = {{aCell.CellPos.x,aCell.CellPos.y+1,aCell.CellPos.z}};
			result.push_back(Cells(idx)) ;
		}
		if(aCell.CellPos.z>0){
			array<a3D_index,3> idx = {{aCell.CellPos.x,aCell.CellPos.y,aCell.CellPos.z-1}};
			result.push_back(Cells(idx)) ;
		}
		if(aCell.CellPos.z<MAXZ-1){
			array<a3D_index,3> idx = {{aCell.CellPos.x,aCell.CellPos.y,aCell.CellPos.z+1}};
			result.push_back(Cells(idx)) ;
		}
			return result;
	}else if(CAmode == CA_MOORE){ //28 rules
		if(aCell.CellPos.x>0)
		if(aCell.CellPos.x>0)
		if(aCell.CellPos.x>0)
			return result;
	}
	return result;
}
void initCells(){

	  for(a3D_index i = 0; i != MAXX; ++i)
	    for(a3D_index j = 0; j != MAXY; ++j)
	      for(a3D_index k = 0; k != MAXZ; ++k)
	        {
	    	  Position temp;
	    	  temp.x = i/MAXX ;
	    	  temp.y = j/MAXY ;
	    	  temp.z = k/MAXZ ;
	    	  Cells[i][j][k].CellPos = temp;
	        }
}

/*
__global__ void stepCell(FloatState *nowState_d, FloatState *nextState_d, canaux *channels_d, int node_number, curandState *devStates)
{

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < node_number)
	{
	    nextState_d[idx] = computeCell(nowState_d, idx, channels_d, devStates);
	}
}

__device__ void computeCell(FloatState *nowState_d, int nodeIndex, canaux *channels_d, curandState* devState_d)
{

}
*/

///////////////Float kernel /////////////////

void initFloat(FloatType *InitFloats, int node_number,int num_starting_float)
{
	int node;

	int startingNode[num_starting_float];

	for (int i = 0; i < node_number; i++)
	{
		InitFloats[i].floatState = NORMAL;
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
	    InitFloats[node].floatState = DRIFT;
	}
}


//-------------------------------------------------------------------------------------------
/**   Version 1.0
***** One cell fired if one of its neighbor is fired. If it fired, it changes to ash. If it is ash, it will become empty.
*****
*/
/*
__device__ FloatState computeStateFloat(FloatState *nowState_d, int nodeIndex, canaux *channels_d, curandState* devState_d)
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
__global__ void stepStateFloat(FloatState *nowState_d, FloatState *nextState_d, canaux *channels_d, int node_number, curandState *devStates)
{

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < node_number)
	{
	    nextState_d[idx] = computeStateFloat(nowState_d, idx, channels_d, devStates);
	}
}
*/

// This method adds constant force vectors to the velocity field
// stored in 'v' according to v(x,t+1) = v(x,t) + dt * f.
__global__ void addForces_k(int dx, int dy, int spx, int spy, float fx, float fy, int r, size_t pitch);

// This method performs the velocity advection step, where we
// trace velocity vectors back in time to update each grid cell.
// That is, v(x,t+1) = v(p(x,-dt),t). Here we perform bilinear
// interpolation in the velocity space.
__global__ void advectVelocity_k(float *vx, float *vy,int dx, int pdx, int dy, float dt, int lb);

// This method performs velocity diffusion and forces mass conservation
// in the frequency domain. The inputs 'vx' and 'vy' are complex-valued
// arrays holding the Fourier coefficients of the velocity field in
// X and Y. Diffusion in this space takes a simple form described as:
// v(k,t) = v(k,t) / (1 + visc * dt * k^2), where visc is the viscosity,
// and k is the wavenumber. The projection step forces the Fourier
// velocity vectors to be orthogonal to the wave wave vectors for each
// wavenumber: v(k,t) = v(k,t) - ((k dot v(k,t) * k) / k^2.     cData *vy,
__global__ void diffuseProject_k( int dx, int dy, float dt,float visc, int lb);

// This method updates the velocity field 'v' using the two complex
// arrays from the previous step: 'vx' and 'vy'. Here we scale the
// real components by 1/(dx*dy) to account for an unnormalized FFT.
__global__ void updateVelocity_k(float *vx, float *vy,int dx, int pdx, int dy, int lb, size_t pitch);

// This method updates the particles by moving particle positions
// according to the velocity field and time step. That is, for each
// particle: p(t+1) = p(t) + dt * v(p(t)).
__global__ void advectParticles_k(int dx, int dy,float dt, int lb, size_t pitch);


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

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource,int modeCA);

const char *sSDKsample = "simpleGL (VBO)";

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void simple_conveyor_kernel(float4 *pos, unsigned int width, unsigned int height,unsigned int mesh_length, int CAMode)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	//z =
    // calculate uv coordinates
    float u = x / (float) width;
    float v = y / (float) height;
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
    	  pos[y*width+x] = make_float4(u, 0.5f, v, 1.0f);

	}else {
		  pos[y*width+x] = make_float4(u, 1.0f, v, 1.0f);

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

    // First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	if (false == initGL(&argc, argv))
	{
		return false;
	}

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
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

	// register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutCloseFunc(cleanup);

	// create VBO
	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
	createVBO(&float_vbo, &float_vbo_cuda_resource, cudaGraphicsMapFlagsWriteDiscard);
	// run the cuda part
	//runCuda(&cuda_vbo_resource,0);
	//runCuda(&float_vbo_cuda_resource,1);
	// start rendering mainloop
	glutMainLoop();

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
    sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
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
    glutCreateWindow("Cuda GL Interop (VBO)");
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

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource,int modeCA)
{
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;

    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
                                                         *vbo_resource));
    //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

    // execute the kernel
	dim3 block(8, 8, 1);
	dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	simple_conveyor_kernel<<< grid, block>>>(dptr, mesh_width, mesh_height,mesh_length, modeCA);

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
    unsigned int size = MAXX * MAXY * MAXZ * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
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

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    runCuda(&cuda_vbo_resource,0);
    runCuda(&float_vbo_cuda_resource,1);

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

    //draw float
    glPointSize(2.0f);
	glBindBuffer(GL_ARRAY_BUFFER, float_vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(0.0, 1.0, 0.0);
	glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
	glDisableClientState(GL_VERTEX_ARRAY);

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
