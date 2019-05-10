/*----------------------------------------------------------------------
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License version 2 as
published by the Free Software Foundation.


Midas Zhou
-----------------------------------------------------------------------*/
#ifndef __NNC_H__
#define __NNC_H__


typedef struct nerve_cell NVCELL; 	/* neuron, or nerve cell */
typedef struct nerve_layer NVLAYER;
typedef struct nerve_net NVNET;

struct nerve_cell
{
	int nin; 	/* number of dendrite receivers */
	NVCELL * const *incells; /* array of input cells, whose outputs are inputs for this cell
				  Only a pointer, will NOT allocate mem space. */
	double *din; 	/* (h^L-1) array of input data, mem space NOT to be allocated. */
	double *dw;  	/* (w) array of weights, mem sapce to be allocated */
	double dv;	/* (b) bias value */
	double dsum;	/* (u) sum of x1*w1+x2*w2+x3*w3+....+xn*wn -dv*/
	double (*transfunc)(double, double, int); /* pointer to a transfer function (also include derivative part) */
	double dout;	/* (h^L) dz, output value, dout=transfunc(dsum-dv) */
	double derr;	/*  SUM(dE/du*w), backfeeded(back propagated) from the next layer cell */
};

struct nerve_layer
{
	int nc;			/* number of NVCELL in the layer */
	NVCELL * *nvcells;	/* array of nerve cells in the layer */
};


struct nerve_net
{
	int nl;			/* number of NVLAYER in the net */
	NVLAYER * *nvlayers;     /*  array of nervers for the net */
};


/* Function declaration */
/* nvcell */
NVCELL * new_nvcell( unsigned int nin, NVCELL * const *incells,
			double *din, double *dw, double bias, double (*transfer)(double, double, int ) );
void free_nvcell(NVCELL *ncell);
int nvcell_rand_dwv(NVCELL *ncell);
int nvcell_feed_forward(NVCELL *nvcell);

int nvcell_feed_backward(NVCELL *nvcell);
int nvcell_input_data(NVCELL *cell, double *data);

/* nvlayers */
NVLAYER *new_nvlayer(unsigned int nc, const NVCELL *template_cell);
void free_nvlayer(NVLAYER *layer);
int nvlayer_feed_forward(NVLAYER *layer);
double nvlayer_mean_loss(NVLAYER *outlayer, const double *tv,
                        double (*loss_func)(double out, const double tv, int token) );
int nvlayer_feed_backward(NVLAYER *layer);

/* nvnet */
NVNET *new_nvnet(unsigned int nl);
int nvnet_feed_forward(NVNET *nnet);
int nvnet_feed_backward(NVNET *nnet);
void free_nvnet(NVNET *nnet);

/* set param */
void  nnc_set_param(double learn_rate);
double random_btwone(void);

/* print params */
void nvcell_print_params(NVCELL *nvcell);
void nvlayer_print_params(NVLAYER *layer);

/* loss func */
double func_lossMSE(double out, const double tv, int token);


#endif
