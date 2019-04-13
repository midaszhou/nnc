/*----------------------------------------------------------------------
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License version 2 as
published by the Free Software Foundation.

Note:
a single neural cell with N inputs:
	1. Input:	 x1,x2,x3,.....xn
	2. Weigths:	 w1,w2,w3,.....wn
	3. valve:	 bias value
	4. sum up:	 u=x1*w1+x2*w2+x3*w3+....+xn*wn -v
	5. transfer function: f(u)
	6. output:	Z=f(u)


Midas Zhou
-----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>


typedef struct nerve_cell NVCELL; 	/* neuron, or nerve cell */
/*
 * 1. If incells==NULL, then it's an input cell.
 */
struct nerve_cell
{
	int nin; 	/* number of dendrite receivers */
	NVCELL *incells; /* input cells, whose outputs are inputs for this cell */
	double *din; 	/* array of input data */
	double *dw;  	/* array of weights */
	double dv;	/* bias value, dsum-=dsum */
	double dsum;	/* sum of x1*w1+x2*w2+x3*w3+....+xn*wn -v */
	double (*transfunc)(double);
	double dout;	/* output value, after transfer */
};

typedef struct nerve_layer NVLAYER;
struct nerve_layer
{
	int ncell;		/* number of NVCELL in the layer */
	NVCELL * *nvcells;	/* array of nerve cells in the layer */
};

typedef struct nerve_net NVNET;
struct nerve_net
{
	int nlayer;		/* number of NVLAYER in the net */
	NVLAYER *nvlayers;       /*  array of nervers for the net */
};



NVCELL * new_ncell( unsigned int nin, const NVCELL *incells,
				double *din, double *dw, double bias, double (*transfer)(double ) );
void free_ncell(NVCELL *ncell);
double transfunc_step(double u);
int feed_forward_nvcell(NVCELL *nvcell);

/*-----------------------------
	MAIN FUNCTION
-----------------------------*/
int main(void)
{
	int i=0;
	double input[5]={1,1,1,1,1};
	double weight[5]={0.2,0.2,0.2,0.2,0.2};

   while(1)
  {
	NVCELL *ncell=new_ncell(5,NULL,input,weight,10,transfunc_step);
	free_ncell(ncell);
	printf("i=%d,\n",i++);
	usleep(5000);
  }

	return 0;
}



/***
 * Params:
 * 	@nin 	number of input data
 * 	@incells 	array of input cells,from which we get input data(din),
 *			if NULL use din as input data, which will be realized only in transfer calculation.
 *	@din	array of input data, if NOT null, use as input data.
 * 	@dw  	array of weights, double. if NULL, init with 0.
 * 	@valve 	valve value.
 * 	@transfer 	transfer function.
 * Return:
 *	pointer to a NCELL ...  OK
 *	NULL		   ...  fails
*/
NVCELL * new_ncell( unsigned int nin, const NVCELL *incells,
				double *din, double *dw, double bias, double (*transfer)(double ) )
{
	int i;

	/* check input param */
	if(nin==0 ) {
		printf("Init a new NCELL: input param 'nin' error.\n");
		return NULL;
	}

	/* calloc ncell->din and dw */
	NVCELL *ncell=calloc(1,sizeof(NVCELL));
	if(ncell==NULL)
		return NULL;
	ncell->din=calloc(2*nin,sizeof(double));
	if(ncell->din==NULL) {
		printf("Init a new NCELL: fail to calloc() din and dw.\n");
		free(ncell);
	}
	ncell->dw=ncell->din+nin; /* din and dw have same size of mem space */

	/* assign din and dw */
	if(din !=NULL) {
		for(i=0;i<nin;i++)
			ncell->din[i]=din[i];
	}
	if(dw !=NULL) {
		for(i=0;i<nin;i++)
			ncell->dw[i]=dw[i];
	}

	/* assign valve and tranfer function */
	ncell->dv=bias;
	ncell->transfunc=transfer;

	return ncell;
}


/***
 * Note:
 *	A step transfer function.
 * Params:
 * 	@u	input param for transfer function;
 * Return:
 *		a double.
*/
void free_ncell(NVCELL *ncell)
{
	if(ncell==NULL) return;

	/* NOTE: din and dw have same size of mem space,
         * so we alloc them together, and free one only!!!
	 */
	if(ncell->din != NULL)
		free(ncell->din);
	//if(ncell->dw != NULL)
	//	free(ncell->dw);
	free(ncell);
	ncell=NULL;
}


/***
 * Note:
 *	A step transfer function.
 * Params:
 * 	@u	input param for transfer function;
 * Return:
 *		a double.
*/
double transfunc_step(double u)
{
	if(u>=0)
		return 1.0;
	else
		return 0.0;
}


/***
 * Note:
 *	A feed forward function for a nerve cell.
 * Params:
 * 	@nvcell		a nerve cell;
 * Return:
 *		0	OK
 *		<0	fails
*/
int feed_forward_nvcell(NVCELL *nvcell)
{
	int i;

	/* check input param */
	if(nvcell==NULL || nvcell->transfunc==NULL)
			return -1;

	/* get input data */
	if(nvcell->incells != NULL) {
		for(i=0; i < nvcell->nin; i++) {
			(nvcell->din)[i]=(nvcell->incells)[i].dout;
		}
	}
	/* else:  use nvcell->din[] as input data */

	/* calculation */
	for(i=0; i < nvcell->nin; i++) {
		nvcell->dsum += nvcell->din[i] * nvcell->dw[i];
	}
	nvcell->dout=nvcell->transfunc(nvcell->dsum - nvcell->dv);

}
