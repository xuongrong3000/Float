/////////////////////////////////
// NGUYEN Ba Diep 13/03/2017   //
// xuongrong3000@gmail.com	   //
// class simpleGL.cu           //
/////////////////////////////////

/*

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <iostream>
// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <timer.h>               // timing functions

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>


//#include "kernel.cu"
#include "defines.h"

#include <cassert>

#define REFRESH_DELAY     20 //ms //200 :slow  10: very fast  prefer:20

int MAXX=64;
int MAXY=64;
int MAXZ=64;

#define SIZE 64

/* **************** global variable for fluid dinamic    2,147,483,648  *****************
#define DT     0.09f     // Delta T for interative solver
#define VIS    0.0025f   // Viscosity constant  //do nhot
#define FORCE (5.8f*DIM) // Force scale factor
#define FR     4         // Force update radius
**************************************************************/

#define IX(i,j,k) ((i)+(M+2)*(j) + (M+2)*(N+2)*(k))
#define MAX(a,b)            (((a) > (b)) ? (a) : (b))

#define IX(i,j,k) ((i)+(M+2)*(j) + (M+2)*(N+2)*(k))
#define SWAP(x0,x) {float * tmp=x0;x0=x;x=tmp;}
#define MAX(a,b)            (((a) > (b)) ? (a) : (b))
#define LINEARSOLVERTIMES 10


extern void dens_step ( int M, int N, int O, float * x, float * x0, float * u, float * v, float * w, float diff, float dt );
extern void vel_step (int M, int N, int O, float * u, float * v,  float * w, float * u0, float * v0, float * w0, float visc, float dt );

//fluid field information
static int M = SIZE; // grid x
static int N = SIZE; // grid y
static int O = SIZE; // grid z
static float dt = 0.4f; // time delta
static float diff = 0.0f; // diffuse
static float visc = 0.0f; // viscosity
static float force = 10.0f;  // added on keypress on an axis
static float source = 200.0f; // density
static float source_alpha =  0.05; //for displaying density

static int addforce[3] = {0, 0, 0};


static float * u, * v, *w, * u_prev, * v_prev, * w_prev;
static float * dens, * dens_prev;

void add_source ( int M, int N, int O, float * x, float * s, float dt )
{
	int i, size=(M+2)*(N+2)*(O+2);
	for ( i=0 ; i<size ; i++ ) x[i] += dt*s[i];
}

void  set_bnd ( int M, int N, int O, int b, float * x )
{
        // bounds are cells at faces of the cube

	int i, j;

	for ( i=1 ; i<=M ; i++ ) {
		for ( j=1 ; j<=N ; j++ ) {
			x[IX(i,j,0 )] = b==3 ? -x[IX(i,j,1)] : x[IX(i,j,1)];
			x[IX(i,j,O+1)] = b==3 ? -x[IX(i,j,O)] : x[IX(i,j,O)];
		}
	}

    for ( i=1 ; i<=N ; i++ ) {
        for ( j=1 ; j<=O ; j++ ) {
            x[IX(0  ,i, j)] = b==1 ? -x[IX(1,i,j)] : x[IX(1,i,j)];
            x[IX(M+1,i, j)] = b==1 ? -x[IX(M,i,j)] : x[IX(M,i,j)];
        }
    }

    for ( i=1 ; i<=M ; i++ ) {
        for ( j=1 ; j<=O ; j++ ) {
            x[IX(i,0,j )] = b==2 ? -x[IX(i,1,j)] : x[IX(i,1,j)];
            x[IX(i,N+1,j)] = b==2 ? -x[IX(i,N,j)] : x[IX(i,N,j)];
        }
    }

    x[IX(0  ,0, 0  )] = 1.0/3.0*(x[IX(1,0,0  )]+x[IX(0  ,1,0)]+x[IX(0 ,0,1)]);
    x[IX(0  ,N+1, 0)] = 1.0/3.0*(x[IX(1,N+1, 0)]+x[IX(0  ,N, 0)] + x[IX(0  ,N+1, 1)]);

    x[IX(M+1,0, 0 )] = 1.0/3.0*(x[IX(M,0,0  )]+x[IX(M+1,1,0)] + x[IX(M+1,0,1)]) ;
    x[IX(M+1,N+1,0)] = 1.0/3.0*(x[IX(M,N+1,0)]+x[IX(M+1,N,0)]+x[IX(M+1,N+1,1)]);

    x[IX(0  ,0, O+1 )] = 1.0/3.0*(x[IX(1,0,O+1  )]+x[IX(0  ,1,O+1)]+x[IX(0 ,0,O)]);
    x[IX(0  ,N+1, O+1)] = 1.0/3.0*(x[IX(1,N+1, O+1)]+x[IX(0  ,N, O+1)] + x[IX(0  ,N+1, O)]);

    x[IX(M+1,0, O+1 )] = 1.0/3.0*(x[IX(M,0,O+1  )]+x[IX(M+1,1,O+1)] + x[IX(M+1,0,O)]) ;
    x[IX(M+1,N+1,O+1)] = 1.0/3.0*(x[IX(M,N+1,O+1)]+x[IX(M+1,N,O+1)]+x[IX(M+1,N+1,O)]);
}

void lin_solve ( int M, int N, int O, int b, float * x, float * x0, float a, float c )
{
	int i, j, k, l;

	// iterate the solver
	for ( l=0 ; l<LINEARSOLVERTIMES ; l++ ) {
		// update for each cell
		for ( i=1 ; i<=M ; i++ ) { for ( j=1 ; j<=N ; j++ ) { for ( k=1 ; k<=O ; k++ ) {
			x[IX(i,j,k)] = (x0[IX(i,j,k)] + a*(x[IX(i-1,j,k)]+x[IX(i+1,j,k)]+x[IX(i,j-1,k)]+x[IX(i,j+1,k)]+x[IX(i,j,k-1)]+x[IX(i,j,k+1)]))/c;
		}}}
        set_bnd ( M, N, O, b, x );
	}
}

void diffuse (  int M, int N, int O, int b, float * x, float * x0, float diff, float dt )
{
	int max = MAX(MAX(M, N), MAX(N, O));
	float a=dt*diff*max*max*max;
	lin_solve ( M, N, O, b, x, x0, a, 1+6*a );
}

void advect (  int M, int N, int O, int b, float * d, float * d0, float * u, float * v, float * w, float dt )
{
	int i, j, k, i0, j0, k0, i1, j1, k1;
	float x, y, z, s0, t0, s1, t1, u1, u0, dtx,dty,dtz;

	dtx=dty=dtz=dt*MAX(MAX(M, N), MAX(N, O));

	for ( i=1 ; i<=M ; i++ ) { for ( j=1 ; j<=N ; j++ ) { for ( k=1 ; k<=O ; k++ ) {
		x = i-dtx*u[IX(i,j,k)]; y = j-dty*v[IX(i,j,k)]; z = k-dtz*w[IX(i,j,k)];
		if (x<0.5f) x=0.5f; if (x>M+0.5f) x=M+0.5f; i0=(int)x; i1=i0+1;
		if (y<0.5f) y=0.5f; if (y>N+0.5f) y=N+0.5f; j0=(int)y; j1=j0+1;
		if (z<0.5f) z=0.5f; if (z>O+0.5f) z=O+0.5f; k0=(int)z; k1=k0+1;

		s1 = x-i0; s0 = 1-s1; t1 = y-j0; t0 = 1-t1; u1 = z-k0; u0 = 1-u1;
		d[IX(i,j,k)] = s0*(t0*u0*d0[IX(i0,j0,k0)]+t1*u0*d0[IX(i0,j1,k0)]+t0*u1*d0[IX(i0,j0,k1)]+t1*u1*d0[IX(i0,j1,k1)])+
			s1*(t0*u0*d0[IX(i1,j0,k0)]+t1*u0*d0[IX(i1,j1,k0)]+t0*u1*d0[IX(i1,j0,k1)]+t1*u1*d0[IX(i1,j1,k1)]);
	}}}

    set_bnd (M, N, O, b, d );
}

void project ( int M, int N, int O, float * u, float * v, float * w, float * p, float * div )
{
	int i, j, k;

	for ( i=1 ; i<=M ; i++ ) { for ( j=1 ; j<=N ; j++ ) { for ( k=1 ; k<=O ; k++ ) {
		div[IX(i,j,k)] = -1.0/3.0*((u[IX(i+1,j,k)]-u[IX(i-1,j,k)])/M+(v[IX(i,j+1,k)]-v[IX(i,j-1,k)])/M+(w[IX(i,j,k+1)]-w[IX(i,j,k-1)])/M);
		p[IX(i,j,k)] = 0;
	}}}

	set_bnd ( M, N, O, 0, div ); set_bnd (M, N, O, 0, p );

	lin_solve ( M, N, O, 0, p, div, 1, 6 );

	for ( i=1 ; i<=M ; i++ ) { for ( j=1 ; j<=N ; j++ ) { for ( k=1 ; k<=O ; k++ ) {
		u[IX(i,j,k)] -= 0.5f*M*(p[IX(i+1,j,k)]-p[IX(i-1,j,k)]);
		v[IX(i,j,k)] -= 0.5f*M*(p[IX(i,j+1,k)]-p[IX(i,j-1,k)]);
		w[IX(i,j,k)] -= 0.5f*M*(p[IX(i,j,k+1)]-p[IX(i,j,k-1)]);
	}}}

	set_bnd (  M, N, O, 1, u ); set_bnd (  M, N, O, 2, v );set_bnd (  M, N, O, 3, w);
}

void dens_step ( int M, int N, int O, float * x, float * x0, float * u, float * v, float * w, float diff, float dt )
{
	add_source ( M, N, O, x, x0, dt );
	SWAP ( x0, x ); diffuse ( M, N, O, 0, x, x0, diff, dt );
	SWAP ( x0, x ); advect ( M, N, O, 0, x, x0, u, v, w, dt );
}

void vel_step ( int M, int N, int O, float * u, float * v,  float * w, float * u0, float * v0, float * w0, float visc, float dt )
{
	add_source ( M, N, O, u, u0, dt ); add_source ( M, N, O, v, v0, dt );add_source ( M, N, O, w, w0, dt );
	SWAP ( u0, u ); diffuse ( M, N, O, 1, u, u0, visc, dt );
	SWAP ( v0, v ); diffuse ( M, N, O, 2, v, v0, visc, dt );
	SWAP ( w0, w ); diffuse ( M, N, O, 3, w, w0, visc, dt );
	project ( M, N, O, u, v, w, u0, v0 );
	SWAP ( u0, u ); SWAP ( v0, v );SWAP ( w0, w );
	advect ( M, N, O, 1, u, u0, u0, v0, w0, dt ); advect ( M, N, O, 2, v, v0, u0, v0, w0, dt );advect ( M, N, O, 3, w, w0, u0, v0, w0, dt );
	project ( M, N, O, u, v, w, u0, v0 );
}

static int allocate_data ( void )
{
	int size = (M+2)*(N+2)*(O+2);

	u			= (float *) malloc ( size*sizeof(float) );
	v			= (float *) malloc ( size*sizeof(float) );
	w			= (float *) malloc ( size*sizeof(float) );
	u_prev		= (float *) malloc ( size*sizeof(float) );
	v_prev		= (float *) malloc ( size*sizeof(float) );
	w_prev		= (float *) malloc ( size*sizeof(float) );
	dens		= (float *) malloc ( size*sizeof(float) );
	dens_prev	= (float *) malloc ( size*sizeof(float) );

	if ( !u || !v || !w || !u_prev || !v_prev || !w_prev || !dens || !dens_prev ) {
		fprintf ( stderr, "cannot allocate data\n" );
		return ( 0 );
	}

	return ( 1 );
}

static void get_force( float * d, float * u, float * v, float * w )
{
	int i, j, k, size=(M+2)*(N+2)*(O+2);;

	for ( i=0 ; i<size ; i++ ) {
		u[i] = v[i] = w[i]= d[i] = 0.0f;
	}

	if(addforce[0]==1) // x
	{
		i=2,
		j=N/2;
		k=O/2;

		if ( i<1 || i>M || j<1 || j>N || k <1 || k>O) return;
		u[IX(i,j,k)] = force*10;
		addforce[0] = 0;
	}

	if(addforce[1]==1)
	{
		i=M/2,
		j=2;
		k=O/2;

		if ( i<1 || i>M || j<1 || j>N || k <1 || k>O) return;
		v[IX(i,j,k)] = force*10;
		addforce[1] = 0;
	}

	if(addforce[2]==1) // y
	{
		i=M/2,
		j=N/2;
		k=2;

		if ( i<1 || i>M || j<1 || j>N || k <1 || k>O) return;
		w[IX(i,j,k)] = force*10;
		addforce[2] = 0;
	}


	return;
}

void sim_main(void)
{
	get_force( dens_prev, u_prev, v_prev, w_prev );
	vel_step ( M,N,O, u, v, w, u_prev, v_prev,w_prev, visc, dt );
	dens_step ( M,N,O, dens, dens_prev, u, v, w, diff, dt );
}


bool show3D = true ;

float g_fAnim = 0.0;
float g_fAnimInc = 0.01f;
bool animFlag = true;
int runControl = 1;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling

float avgFPS = 0.0f;
unsigned int frameCount = 0;


GLint gFramesPerSecond = 0;
static GLint Frames = 0;         // frames averaged over 1000mS
  static GLuint Clock;             // [milliSeconds]
  static GLuint PreviousClock = 0; // [milliSeconds]
  static GLuint NextClock = 0;     // [milliSeconds]

void FPS(void) {

  ++Frames;
  Clock = glutGet(GLUT_ELAPSED_TIME); //has limited resolution, so average over 1000mS
  if ( Clock < NextClock ) return;

  gFramesPerSecond = Frames/1; // store the averaged number of frames per second

  PreviousClock = Clock;
  NextClock = Clock+1000; // 1000mS=1S in the future
  Frames=0;
}

void timerMeasure(int value)
{
  const int desiredFPS=100;
  glutTimerFunc(1000/desiredFPS, timerMeasure, ++value);

  //put your specific idle code here
  //... this code will run at desiredFPS

 // printf("numframe:%d  ",Frames);
  //end your specific idle code here

  FPS(); //only call once per frame loop to measure FPS
  glutPostRedisplay();
}


// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

StopWatchInterface *timer = NULL;


int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

GLuint float_vbo;
float4 *floatPos;
/*
struct cudaGraphicsResource *float_vbo_cuda_resource;
void *d_float_vbo_buffer = NULL;
*/
////////////////// bien toan cuc luu tru thong tin

bool showFloat = true;
int num_floats = 4;
FloatType *AllFloats_host = NULL;
FloatType *AllFloats_device;

bool showCell = true;
CellType *AllCells_host = NULL;
CellType *AllCells_device;

Index *cell_index_host = NULL;
Index *cell_index_device;

float4 *surfacePos;
GLuint surfaceVBO;
bool showSurface = true;

float *floatcolorred;
float *floatcolorgreen;
float *floatcolorblue;


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);
extern __global__ void game_of_life_kernel(float4 *pos, unsigned int maxx,unsigned int maxy, unsigned int maxz, int CAMode, CellType *Cells_device,Index * index_device,bool showMode);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);
void computeFPS();
// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource,int modeCA,CellType *cells_d,Index * index_device);


void initCell2D(int CAMode){
	long tempid = 0;
	int num_inactive = 0;
	    for(int j = 0; j < MAXY; ++j){
	        for(int i = 0; i < MAXX; ++i)
	        {
	    	  Position temp;
	    	  temp.x = (float)i/MAXX ;
	    	  temp.y = (float)j/MAXY ;
	    	  temp.z = 0.5f;
	    	  unsigned long long int index = i+MAXY*j;
	    	  AllCells_host[index].id = tempid;
	    	  AllCells_host[index].CellPos = temp;
	    	  int state  = rand() % 100 ;
	//    	  cout << " state = " <<state;
	    	  if (state %4 ==0) { //Diep random init
	    		  AllCells_host[index].state = NORMAL ;
	    	//	  cout << " \n NORMAL id = " <<tempid;
	    	  }else {
	    		  AllCells_host[index].state = INACTIVE ;
	    	//	  cout << " \n INACTIVE id = " <<tempid;
	    		  num_inactive ++;
	    	  }
	    	  tempid++;
	        }
	    }
	   // cout << " tempid = " <<tempid;
	    if(CAMode==CA_VON_NEUMANN){ //4 neighbor 2D
	    	vector<long> neighbor ;
	    	for(int j = 0; j < MAXY; ++j){
			   for(int i = 0; i < MAXX; ++i)
				{  unsigned long long int index = i+MAXY*j;
				   long tempindex[NUM_NEIGHBOR];

				   if (i>0){//left(x) = (x - 1) % M
					   tempindex[0] = AllCells_host[index-1].id ;
				   }else {
					   tempindex[0] = INVALID_ID ;
				   }
				   if (i<MAXX-1){//right(x) = (x + 1) % M
					   tempindex[1] = AllCells_host[index+1].id ;
				   }else{
					   tempindex[1] = INVALID_ID ;
				   }
				   if (j>0){//above(x) = (x - M) % (M * N)
					   tempindex[2] = AllCells_host[index-MAXX].id ;
				   }else {
					   tempindex[2] = INVALID_ID ;
				   }
				   if (j<MAXY-1){//below(x) = (x + M) % (M * N)
					   tempindex[3] = AllCells_host[index+MAXX].id ;
				   }else {
					   tempindex[3] = INVALID_ID ;
				   }
				   memcpy(cell_index_host[index].id, tempindex, NUM_NEIGHBOR * sizeof(long)); //CA Diep change size
				//   cell_index_host[i+(j*MAXX)].id = tempindex;

		/*		   if(i==2&&j==0){
					   cout << "\n i+j*MAXX= " << i+j*MAXX << " AllCells id= " <<AllCells[(i+j*MAXX)].id << " neightbors: "
							   << AllCells[((i+j*MAXX)-1)%MAXX].id <<","<< AllCells[((i+j*MAXX)+1)%MAXX].id <<","
							   << AllCells[((i+j*MAXX)-MAXX)%(MAXX*MAXY)].id <<","<< AllCells[((i+j*MAXX)+MAXX)%(MAXX*MAXY)].id;
				   }*/
				}
	    	}
	    }
	printf("\n done initCell maxid = %d , inactive=%d ",tempid,num_inactive);
}

void initCell3D(int CAMode){
	long tempid = 0;
	int num_inactive = 0;
	for(int k=0;k<MAXZ;k++){
	    for(int j = 0; j < MAXY; j++){
	        for(int i = 0; i < MAXX; i++)
	        { unsigned long long int index = i+MAXZ*(j+MAXY*k);
	    	  Position temp;
	    	  temp.x = (float)i/MAXX ;
	    	  temp.y = (float)j/MAXY ;
	    	  temp.z = (float)k/MAXZ ;

	    	  AllCells_host[index].id = tempid;
	    	  AllCells_host[index].CellPos = temp;
	    	  int state  = rand() % 200 ;
	    //	  cout << " \ni+MAXZ*(j+MAXY*k)=  " <<i+MAXZ*(j+MAXY*k) << " tempid="<<tempid;
	    	  if (state %4 ==0) { //Diep random init
	    		  AllCells_host[index].state = NORMAL ;
	    	//	  cout << " \n NORMAL id = " <<tempid;
	    	  }else {
	    		  AllCells_host[index].state = INACTIVE ;
	    	//	  cout << " \n INACTIVE id = " <<tempid;
	    		  num_inactive ++;
	    	  }
	    	  tempid++;
	    	  //Flat[x + HEIGHT* (y + WIDTH* z)]
//	    	  The algorithm is mostly the same. If you have a 3D array Original[HEIGHT, WIDTH, DEPTH] then you could turn it into Flat[HEIGHT * WIDTH * DEPTH] by
//	    	  Flat[x + WIDTH * (y + DEPTH * z)] = Original[x, y, z]

			}//end for i MAXX
    	}//end for j MAXY
	}//end for k MAXZ

	//    cout << " tempid = " <<tempid;
	    if(CAMode==CA_VON_NEUMANN){ //6 neighbor 3D
	    	for (int k=0; k<MAXZ;k++){
				for(int j = 0; j < MAXY; j++){
				   for(int i = 0; i < MAXX; i++){

					   long tempindex[NUM_NEIGHBOR];

					   if (i>0){//left(x) = (x - 1) % M
						   tempindex[0] = AllCells_host[((i+MAXZ*(j+MAXY*k))-1)].id ;
					   }else {
						   tempindex[0] = INVALID_ID ;
					   }
					   if (i<MAXX-1){//right(x) = (x + 1) % M
						   tempindex[1] = AllCells_host[((i+MAXZ*(j+MAXY*k))+1)].id ;
					   }else{
						   tempindex[1] = INVALID_ID ;
					   }
					   if (j>0){//above(x) = (x - M) % (M * N)
						   tempindex[2] = AllCells_host[((i+MAXZ*(j-1+MAXY*k)))].id ;
					   }else {
						   tempindex[2] = INVALID_ID ;
					   }
					   if (j<MAXY-1){//below(x) = (x + M) % (M * N)
						   tempindex[3] = AllCells_host[((i+MAXZ*(j+1+MAXY*k)))].id ;
					   }else {
						   tempindex[3] = INVALID_ID ;
					   }
					   if (k>0){//behind (x) = (x - M) % (M * N)
						   tempindex[4] = AllCells_host[(i+MAXZ*(j+MAXY*(k-1)))].id ;
					   }else {
						   tempindex[4] = INVALID_ID ;
					   }
					   if (k<MAXZ-1){//front (x) = (x + M) % (M * N)
						   tempindex[5] = AllCells_host[(i+MAXZ*(j+MAXY*(k+1)))].id ;
					   }else {
						   tempindex[5] = INVALID_ID ;
					   }

					   memcpy(cell_index_host[i+MAXZ*(j+MAXY*k)].id, tempindex, NUM_NEIGHBOR * sizeof(long)); //CA Diep change size
					//   cell_index_host[i+(j*MAXX)].id = tempindex;

					 //  if(i==0&&j==1&&k==1){
					/*	   cout <<"\n "<<k<<j<<i <<"|i+MAXZ*(j+MAXY*k)= " << i+MAXZ*(j+MAXY*k) << " AllCells id= " <<AllCells_host[i+MAXZ*(j+MAXY*k)].id << " \n neightbors: ";
						   for (int de=0;de<NUM_NEIGHBOR;de++){
							   cout << de << ":"<< tempindex[de]<< " |" ;
						   }*/
					//   }
				/*	   if(i==1&&j==1&&k==1){
						   cout << "\ni+MAXZ*(j+MAXY*k)= " << i+i+MAXZ*(j+MAXY*k) << " AllCells id= " <<AllCells_host[i+MAXZ*(j+MAXY*k)].id << " \n neightbors: ";
						   for (int de=0;de<NUM_NEIGHBOR;de++){
							   cout << de << ":"<< tempindex[de]<< " |" ;
						   }
					   }
				*/
					}//end for i MAXX
				}//end for j MAXY
	    	}//end for k MAXZ
	    }//end if CA Mode

//	printf("\n done initCell maxid = %d , inactive=%d ",tempid,num_inactive);
}

void initFloat(){
	floatcolorred = (float *)malloc(num_floats*sizeof(float));
	floatcolorgreen =(float *)malloc(num_floats*sizeof(float));
	floatcolorblue = (float *)malloc(num_floats*sizeof(float));
	for (int k=0;k<num_floats; k++){
		FloatType tempfloattype;
		tempfloattype.trajectory = (FloatTrajectoryPoint*)malloc (MAX_TRAJECTORY_SIZE *sizeof(FloatTrajectoryPoint));
		tempfloattype.trajectory_size = MAX_TRAJECTORY_SIZE ;
		for(int j = 0; j < MAX_TRAJECTORY_SIZE; ++j){
		    FloatTrajectoryPoint temppoint;
		    temppoint.measure = (FloatMeasurement *) malloc (MAX_MEASURE_SIZE*sizeof(FloatMeasurement));
		    temppoint.measure_size = MAX_MEASURE_SIZE;
		    for(int i = 0; i < MAX_MEASURE_SIZE; ++i)
		    {
			    FloatMeasurement tempmes;
			    tempmes.pressure = (float)(rand() % 200)*10;
			    tempmes.salinity = (float)(rand() % 360)/10;
			    tempmes.temperature = (float)(rand() % 360)/10;
			    temppoint.measure[i] = tempmes;
		    }
		    Position temppos;
		    temppos.x = (float)(rand() % MAXX)/MAXX ;
		    temppos.y = (float)(rand() % MAXY)/MAXY ;
		    temppos.z = (float)(rand() % MAXZ)/MAXZ ;

		////////////////////
		//   add date  /////
		////////////////////

		    temppoint.FloatPos = temppos;
		    tempfloattype.trajectory[j] = temppoint;
	    }
	    tempfloattype.id =k ;
	    tempfloattype.floatState = DRIFT;
	    AllFloats_host[k] = tempfloattype;
	//  memcpy(cell_index_host[i+(j*MAXX)].id, tempindex, NUM_NEIGHBOR * sizeof(long)); //CA Diep change size
	    floatcolorred[k] = (float)(rand()%100)/100;
	    floatcolorblue[k]  = (float)(rand()%100)/100;
	    floatcolorgreen[k]  = (float)(rand()%100)/100;
	}
}

void initSurface(){
	surfacePos = (float4 *) malloc(sizeof(float4)*MAXX*MAXY);
	for (int j=0; j<MAXY; j++){
		for (int i=0; i<MAXX;i++){
			float x = (float) i/MAXX ;
			float y = (float) j/MAXY ;
			surfacePos[j*MAXX+i] = make_float4(x, 1.0f, y, 1.0f);
		}
	}

//	 assert(surfaceVBO);
	// create buffer object

/*
	GLuint points_vbo = 0;
	glGenBuffers(1, &points_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, points_vbo);
	glBufferData(GL_ARRAY_BUFFER, 9 * sizeof(float), points, GL_STATIC_DRAW);
*/

/*
	glGenBuffers(1, VertexVBOID);
	  glBindBuffer(GL_ARRAY_BUFFER, VertexVBOID);
	  glBufferData(GL_ARRAY_BUFFER, sizeof(MyVertex)*3, &pvertex[0].x, GL_STATIC_DRAW);

	  ushort pindices[3];
	  pindices[0] = 0;
	  pindices[1] = 1;
	  pindices[2] = 2;

	  glGenBuffers(1, &IndexVBOID);
	  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexVBOID);
	  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ushort)*3, pindices, GL_STATIC_DRAW);
*/
}


void runCudaLoop(int valueControl){
//	cout << " value " << ":"<< valueControl;
	if (valueControl==0){
		//do nothing
	}else{
		runCuda(&cuda_vbo_resource,0,AllCells_device,cell_index_device);
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


	int arraycellsize = MAXX*MAXY*MAXZ*sizeof(CellType);
	int arrayindex = MAXX*MAXY*MAXZ*sizeof(Index);
	int arrayfloatsize = num_floats*sizeof(FloatType);
		//Allocating memory of host variables
	AllCells_host = (CellType*) malloc(arraycellsize);
	cell_index_host = (Index*) malloc(arrayindex);
	AllFloats_host = (FloatType *) malloc(arrayfloatsize);
		//Allocating memory to device variable
	if(show3D){
		initCell3D(CA_VON_NEUMANN);
	}else{
		initCell2D(CA_VON_NEUMANN);
	}
	//
	if(showSurface){
		initSurface();
	}

	if(showFloat){
		initFloat();
	}

//	int arraycellsize = MAXX*MAXY*sizeof(CellType);
	checkCudaErrors(cudaMalloc((CellType**)&AllCells_device,arraycellsize));
	checkCudaErrors(cudaMemcpy(AllCells_device, AllCells_host, arraycellsize, cudaMemcpyHostToDevice));

//	int arrayindex = MAXX*MAXY*sizeof(Index);
	checkCudaErrors(cudaMalloc(( Index** ) &cell_index_device,arrayindex));
	checkCudaErrors(cudaMemcpy(cell_index_device, cell_index_host, arrayindex, cudaMemcpyHostToDevice));

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

//	glutTimerFunc(0,timerMeasure,0);

	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

	glutMainLoop();

}


////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource,int modeCA,CellType *Cells_device,Index *index_device)
{
    float4 *dptr;

    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
                                                         *vbo_resource));
    dim3 block(8, 8, 1);
	dim3 grid(MAXX / block.x, MAXY / block.y, MAXZ/block.z);
	game_of_life_kernel<<< grid, block>>>(dptr, MAXX,MAXY,MAXZ, modeCA,Cells_device,index_device,show3D);

    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(vbo);

    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    unsigned int size = MAXX * MAXY * MAXZ  * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
//	cout<<"\n average time = "<<sdkGetAverageTimerValue(&timer) / 1000.f ;
/*
   if (sdkGetAverageTimerValue(&timer)>0.1) {
		sdkStopTimer(&timer);sdkStartTimer(&timer);
		return;
	}
*/

    sdkStartTimer(&timer);

	glutTimerFunc(1000/g_fAnim, runCudaLoop, runControl);

    // run CUDA kernel to generate vertex positions
  //  runCuda(&cuda_vbo_resource,0,AllCells_device,cell_index_device);

  //  cudaDeviceSynchronize();
 //   checkCudaErrors(cudaMemcpy(AllCells_host,AllCells_device, arraycellsize, cudaMemcpyDeviceToHost));

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    if(showCell) {
		glPointSize(3.0f);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glVertexPointer(4, GL_FLOAT, 0, 0);

		glEnableClientState(GL_VERTEX_ARRAY);
		glColor4f(1.0, 0.0, 0.0,0.5f);
		glDrawArrays(GL_POINTS, 0, MAXX*MAXY*MAXZ);
		glDisableClientState(GL_VERTEX_ARRAY);
    }
    if(showSurface){
    	glGenBuffers(1, &surfaceVBO);
		glBindBuffer(GL_ARRAY_BUFFER, surfaceVBO);
		unsigned int size = MAXX * MAXY  * 4 * sizeof(float);
		glBufferData(GL_ARRAY_BUFFER, size, surfacePos, GL_STATIC_DRAW);

   // 	glBindBuffer(GL_ARRAY_BUFFER, surfaceVBO);
		glVertexPointer(4, GL_FLOAT, 0, 0);

		glEnableClientState(GL_VERTEX_ARRAY);
		glColor4f(0.0, 0.0, 1.0f,1.0f);
		glDrawArrays(GL_POINTS, 0, MAXX*MAXY);
		glDisableClientState(GL_VERTEX_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    if(showFloat){
    	GLuint float_vbo;
    	float4 *floatPos;
    	for(int k=0;k<num_floats;k++){
			glGenBuffers(1, &float_vbo);
			glBindBuffer(GL_ARRAY_BUFFER, float_vbo);
			unsigned int trajecsize = AllFloats_host[k].trajectory_size  * 4 * sizeof(float);
			floatPos = (float4*) malloc (trajecsize);
			for(int i =0; i<AllFloats_host[k].trajectory_size;i++){
				floatPos[i] = make_float4(AllFloats_host[k].trajectory[i].FloatPos.x, AllFloats_host[k].trajectory[i].FloatPos.z, AllFloats_host[k].trajectory[i].FloatPos.y, 1.0f);
			}
			glBufferData(GL_ARRAY_BUFFER, trajecsize, floatPos, GL_STATIC_DRAW);
	   // 	glBindBuffer(GL_ARRAY_BUFFER, surfaceVBO);
			glVertexPointer(4, GL_FLOAT, 0, 0);
			glEnableClientState(GL_VERTEX_ARRAY);
			glColor4f(floatcolorred[k] , floatcolorgreen[k] , floatcolorblue[k] ,1.0f);
			glDrawArrays(GL_LINE_STRIP, 0, AllFloats_host[k].trajectory_size);
		//	void glutWireSphere(GLdouble radius, GLint slices, GLint stacks);
			glDisableClientState(GL_VERTEX_ARRAY);

			glBindBuffer(GL_ARRAY_BUFFER, 0);
    	}
    }
  //  glColor3f(1.0,0.0,0.0);
  //	glLoadIdentity();
 //	glutWireSphere( 0.05, 8, 4);
 //	glFlush();
    glutSwapBuffers();

    g_fAnim += g_fAnimInc;
    if(animFlag) {
  //      glutPostRedisplay();
    }
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

    cudaDeviceReset();

    cudaFree(AllFloats_device);
	cudaFree(AllCells_device);
	cudaFree(cell_index_device);

	free(AllFloats_host);
	free(AllCells_host);
	free(cell_index_host);
	free(floatPos);
	free(surfacePos);
	free(floatcolorred);
	free(floatcolorgreen);
	free(floatcolorblue);
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

        case 'a': // toggle animation
	    case 'A':
            animFlag = (animFlag)?0:1;
            break;
	    case '-': // decrease the time increment for the CUDA kernel
            g_fAnimInc -= 0.01;
            break;
	    case '+': // increase the time increment for the CUDA kernel
	    	g_fAnimInc += 0.01;
            break;
	    case 'r': // reset the time increment
	    	g_fAnimInc = 0.01;
            break;
	    case 'f':showFloat = !showFloat;
	    	break;
	    case 's':
	    case 'S':
	    	runControl ++;
	    	runControl %=2;
	    	break;
	    case 'z':
	    	showSurface = !showSurface;
	    	break;
    }
     glutPostRedisplay();
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

    glClearColor(0.0, 0.0, 0.0, 1.0);

    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);

    glEnable(GL_BLEND); //enable alpha color
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);//enable alpha color

    glViewport(0, 0, window_width, window_height);

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
