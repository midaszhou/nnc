/*----------------------------------------------------------------------
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License version 2 as
published by the Free Software Foundation.


Midas Zhou
-----------------------------------------------------------------------*/
#ifndef __NNC_H__
#define __NNC_H__


#define DERIVATIVE_FUNC	1  /* to switch to derivative calculation in a function */
#define NORMAL_FUNC	0


typedef struct nerve_cell NVCELL; 	/* neuron, or nerve cell */
typedef struct nerve_layer NVLAYER;
typedef struct nerve_net NVNET;

struct nerve_cell
{
	int nin; 	/* number of dendrite receivers */
	NVCELL * const *incells; /* array of input cells, whose outputs are inputs for this cell
				  Only a pointer, will NOT allocate mem space. */
	double *din; 	/* array of input data, mem space NOT to be allocated. */
	double *dw;  	/* array of weights, mem sapce to be allocated */
	double dv;	/* bias value, dsum - dv */
	double dsum;	/* dh, sum of x1*w1+x2*w2+x3*w3+....+xn*wn */
	double (*transfunc)(double,int); /* pointer to a transfer function (also include derivative part) */
	double dout;	/* dz, output value, dout=transfunc(dsum-dv) */
	double derr;	/* error(loss) or error(loss) value back feeded(back propagated) from the next layer cell */
};

struct nerve_layer
{
	int nc;		/* number of NVCELL in the layer */
	NVCELL * *nvcells;	/* array of nerve cells in the layer */
};


struct nerve_net
{
	int nlayer;		/* number of NVLAYER in the net */
	NVLAYER *nvlayers;       /*  array of nervers for the net */
};


/* Function declaration */
NVCELL * new_nvcell( unsigned int nin, NVCELL * const *incells,
				double *din, double *dw, double bias, double (*transfer)(double, int ) );
void free_nvcell(NVCELL *ncell);
int nvcell_rand_dwv(NVCELL *ncell);
int nvcell_feed_forward(NVCELL *nvcell);
int nvcell_feed_backward(NVCELL *nvcell, const double *tv);
int nvcell_input_data(NVCELL *cell, double *data);


NVLAYER *new_nvlayer(int nc, const NVCELL *template_cell);
void free_nvlayer(NVLAYER *layer);
int nvlayer_feed_forward(NVLAYER *layer);
int nvlayer_feed_backward(NVLAYER *layer, double *tv);


void  nnc_set_param(double learn_rate);
double func_step(double u, int token);
double func_sigmoid(double u, int token);
double random_btwone(void);


#endif
