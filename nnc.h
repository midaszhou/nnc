/*----------------------------------------------------------------------
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License version 2 as
published by the Free Software Foundation.


Midas Zhou
-----------------------------------------------------------------------*/
#ifndef __NNC_H__
#define __NNC_H__

#include <stdint.h>
#include <stdbool.h>

typedef struct nerve_cell  NVCELL; 	/* neuron, or nerve cell */
typedef struct nerve_layer NVLAYER;
typedef struct nerve_net   NVNET;

typedef struct conv3x3	   CONV3X3;
typedef struct maxpool2x2  MAXPOOL2X2;

struct nerve_cell
{

 /* ---- TODO : For Convolution NVCELL : fs>0  ----- */
	unsigned int fs; 		/* Filter size 3,5,7 etc. fs*fs is number of dendrite receivers */
	int iw,ih;			/* Input data size */
	NVCELL **pincells;		/* This is for Convolution nvCell, allocated in new_conv_nvcell(). HK2023-07-08
					 * A list of NVELL*, map to connected/input nvcells
					 * pincells[iw*ih]
					 */
	double **pdin;			/* This is for Convolution nvCell, allocated in new_conv_nvcell(). HK2023-07-08
					 * A list of double *, map to receiving data.
					 * pdin[iw*ih]
					 */
	double *douts;			/* Output values, size (iw-fs+1)*)ih-fs+1)*/



 /* ---- For Non Convolution NVCELL ----- */
	unsigned int nin; 		/* number of dendrite receivers */

	NVCELL * const *incells; 	/* array of input cells, whose outputs are inputs for this cell
				  	   Only a reference pointer here.
					 */
	double *din; 			/* (data*nin)  (h^L-1) array of input data.
					 * Only a reference pointer here. (mem space NOT to be allocated here.)
					 *  If the cell is in the input layer, din points to input data.
					 *  Else NO USE!  data fetch from incells[]->dout
					 */
	double *prederr;		/* as flattened prev. &MAXPOOL->derr[0][0] */


  /* ----- For Common NVCELLs ----- */

	double *dw;  			/* (w*nin) array of weights, mem sapce allocated in new_nvcell().  */

	bool   ignore_dv;		/* Ignore bias, For example: a convolution ncell. HK2023-07-08 */
	double dv;			/* (b) bias value */

	/* Pooling operations: like Max(), Min(), or Average(). HK2023-07-08 */
	double (*pool)(double*, int);

	double dsum;			/* (u) sum of x1*w1+x2*w2+x3*w3+....+xn*wn -dv */

	double (*transfunc)(double, double, int); /* pointer to a NVCELL transfer function (also include derivative part)
						   * If NULL, then f(x)=x! as nvcell->dout=nvcell->dsum <-------- NOTICED~!!!

									!!!---- CAUTION ----!!!
						      If the layer has transfunc defined(softMax etc.), then all
 						      layer->nvcells SHOULD NOT have transfunc defined!
						      If both transfuncs are defined(both nonlinear functions), usually
						      learning process will fail (NOT converge)!
						   */

	double dout;			/* (h^L) dz, output value, dout=transfunc(dsum)
					 * If transfunc==NULL, then dout==dsum!!!   see in nvcell_feed_forward()
					 */

	double derr;			/* current nvcell's dE/du, updated in feedback process.
					  1. In backpropagation, it temporarily stores dE/dh(=next layer' dE/dxi). then dE/du=dE/dh*f'(u).
			   		  2. For non_output layer, after feedback calculation:
				derr=dE/du=f'(u)*SUM(dE/du*w), wereh SUM(dE/du*w)=backfeeded(back propagated) from the next layer cell
			   		  3. For output layer, after feedback calculation:
					     derr=dE/du=f'(u)*L'(h)
					  4. Reset at nvnet_feed_backward() before feeding backward.
					*/
};

/*-------------------------------------------------------
Note:
1. Padding mode:
   full-padding, same-padding, valid-padding.
2. Stride=S, FilterSize=F,  ImageSize=WxH
   then for valid_padding, out put image size:
   OW=(W-F+1)/S; OH=(H-F+1)/S

*------------------------------------------------------*/
struct conv3x3
{
	unsigned int nf;	/* Number of filters */
	//unsigned int stride;  /* ALWAYS ==1, Stride */
	//unsigned int chans;	/* ALWAYS ==1, Channels */
	//mode=valid_padding;

	double** fparams;	/* Array of all filter parameters/data, nf*9*sizeof(double), calloc in new_conv3x3()
				 * fparams[filter_index][param_index(0 ~ 3*3-1)]
				 * Data in row_major.
				 * Also see conv3x3_rand_params()
				 */

	double **dFP;		/* dE/dFP(Filter_Parameter),  dFP[filter_index][param_index(0 ~ (3*3-1))]    HK2023-07-11
				 * This for updating fparams: fparams[nf][x] += -learRate*dFP[nf][x]
				 * XXX It SHOULD be updated after each backfeed opeartion, before updating fparams.
				 * Updated in conv3x3_feed_backward().
				 */

	unsigned int imw,imh;	/* Original(Input) image(or other data) width/height */
	//unsigned int fs==3;	/* Filter size, 3,5,7 etc. */

	unsigned int ow,oh;	/* w,h for douts, w=imw-fs+1, h=imh-fs+1, (imw-2,imh-2) */

	double *din;		/* Pointer to input image data, size imw*imh, row-major
				 * If connects to a MAXPOOL2x2 outputs, it should be flattened data (pointer).
				 */
	double *prederr;	/* For feeding back derr, Example: flattened prev. &maxpool->derr[0][0]. */
			        /* Array size MUST be same as din */

	double **douts;		/* Convolution output, size= nf*(imw-2)*(imh-2)
				 * Stored as results of the last forward operations, for next backprop operation.
				 * douts[filter_index][0 ~ (imw-2)*(imh-2)-1], row-major
				 */
	/* -------> Flatted douts:  (double *)(&douts[0][0]) */

	double **derr;		/* dE/du dLoss/dOut  derr[filter_index][0 ~ (imw-2)*(imh-2)-1]
				   1. In backpropagation, it temporarily stores dE/dh(=next layer' dE/dxi).
				      dE/du=dE/dh*f'(u)=dE/dh, here f(u)=u, f'(u)=1.
				   2. Reset at nvnet_feed_backward(), before feeding backward.
				 */
};


struct maxpool2x2
{
	CONV3X3* inconv3x3; 	/* Pointer to input CONV3X3, whose outputs are inputs for this cell
				    Only a reference pointer here.
				 */

	unsigned int nf;	/* Number of filters, USUALLY nf == inconv3x3->nf  */

	#if 0 ////// NO NEED /////
	double** fparams;	/* Array of all filter parameters/data, nf*4*sizeof(double), calloc in new_maxpool2x2()
				 * fparams[filter_index][param_index(0 ~ 3)]
				 */
	#endif

	unsigned int imw, imh;	/* Input data size width/height */
	unsigned int ow, oh;	/* data width/height, w=inw//2, h=inh//2, or  w(h)=(inconv3x3->imw(imh)-2)//2 */

	double **din;		/* Pointer to input data, size nf*inw*inh,  USUALLY as conv3x3->douts[][] */
//	double *prederr;	/* For feeding back derr, Example: flattened prev. conv3x3->derr */
			        /* Array size MUST be same as din */

	double **douts;		/* maxpool2x2 output, size= w * h *nf, in row-major
				 * Stored as results of the last forward operations, for next backprop operation.
				 * douts[filter_index][data_index(0 ~ w*h-1)]
				 */
	/* -------> Flatted douts:  (double *)(&douts[0][0]) */

	double **derr;		/* dE/du  dLoss/dOut derr[filter_index][0 ~ ow*oh-1]
				   1. In backpropagation, it temporarily stores dE/dh(=next layer' dE/dxi).
				      dE/du=dE/dh*f'(u)=dE/dh, here f(u)=u; f'(u)=1.
				   2. Reset at nvnet_feed_backward(), before feeding backward.
				 */
};


struct nerve_layer
{
/* : ConvLayer, NeurionLayer, PoolLayer */

	/* ------- For Convolution Layer --------- */
	CONV3X3	*conv3x3;	 /* Pointer to a 3x3 convolution layer */
	MAXPOOL2X2 *maxpool2x2;	 /* Pointer to a 2x2 max pooling layer */


	/* ------- For Neurion Layer ------- */
	unsigned int nc;	/* number of NVCELL in the layer */

	NVCELL **nvcells;	/* array of nerve cells(pointers) in the layer, calloc in new_nvlayer()  */

	int (*transfunc)(NVLAYER*, int);  /* pointer to a NVLAYER transfer function (also include derivative part)
                                            * NOW: It's ONLY for the output layer, For example, to apply softMax function.
                                            * If NULL, then f(x)=x! as nvcell->dout=nvcell->dsum

								!!!---- CAUTION ----!!!
					      If the layer has transfunc defined(softMax etc.), then all
 					      layer->nvcells SHOULD NOT have transfunc defined!
					      If both transfuncs are defined(both nonlinear functions), usually
					      learning process will fail!

                                            */
	double *douts;		/* For the output layer, to store transfunc results! HK2023-06-19
				 * In this case, nvcell->dout stores u value.
				 * Calloc in new_nvlayer()
				 */
};


struct nerve_net
{
	unsigned int nl;			/* number of NVLAYER in the net */
	NVLAYER * *nvlayers;     /*  array of nervers for the net, calloc in new_nvnet(). */

	unsigned long np;	/* total numbers of params in the net */
	double *params;		/* for buffing params of all cells in the nvnet,
				 * WARNING: write and read MUST follow the same sequence!!!
				 * params are buffed cell by cell, in order of: dw[], dv, dsum, dout, derr
				*/
	unsigned long nmp;	/* total nmbers of mmts of all params, dw[] and dv */
	double *mmts; 		/* momentums of all corresponding params
				 * WARNING: write and read MUST follow the same sequence!!!
				 */
};


/* Function declaration */
/* nvcell */
NVCELL *new_nvcell( unsigned int nin, NVCELL * const *incells,
			double *din, double *dw, double bias, double (*transfer)(double, double, int ) );
void free_nvcell(NVCELL *ncell);
int nvcell_rand_dwv(NVCELL *ncell);
int nvcell_feed_forward(NVCELL *nvcell);
int nvcell_feed_backward(NVCELL *nvcell);
int nvcell_input_data(NVCELL *cell, double *data);

/* conv3x3 */
CONV3X3  *new_conv3x3(unsigned int numFilters, unsigned int w, unsigned int h, double *din);
void free_conv3x3(CONV3X3 *conv3);
int conv3x3_rand_params(CONV3X3 *conv3);
int conv3x3_feed_forward(CONV3X3 *conv3);
int conv3x3_feed_backward(CONV3X3 *conv3);

/* maxpool2x2 */
MAXPOOL2X2  *new_maxpool2x2( CONV3X3 *pinconv3x3, unsigned int numFilters, unsigned int imw, unsigned int imh,  double **din);
void free_maxpool2x2(MAXPOOL2X2 *maxpool);
//int maxpool2x2_rand_params(MAXPOOL2X2 *maxpool);
int maxpool2x2_feed_forward(MAXPOOL2X2 *maxpool);
int maxpool2x2_feed_backward(MAXPOOL2X2 *maxpool);

/* nvlayers */
NVLAYER *new_nvlayer(unsigned int nc, const NVCELL *template_cell, bool layerTransfuncDefined);
NVLAYER *new_conv_nvlayer( NVCELL * const *incells, double *din, unsigned int iw, unsigned ih, unsigned int nf, unsigned int fs);
void free_nvlayer(NVLAYER *layer);
int nvlayer_load_params(NVLAYER *layer, double *weights, double *bias);
int nvlayer_link_inputdata(NVLAYER *layer, double *data);
int nvlayer_feed_forward(NVLAYER *layer);
double nvlayer_mean_loss(NVLAYER *outlayer, const double *tv,
                        double (*loss_func)(double out, const double tv, int token) );
int nvlayer_feed_backward(NVLAYER *layer);


/* nvnet */
NVNET *new_nvnet(unsigned int nl);
//int nvnet_feed_forward(NVNET *nnet);
int nvnet_init_params(NVNET *nnet);
double nvnet_feed_forward(NVNET *nnet, const double *tv,
                          double (*loss_func)(double, const double, int) );
int nvnet_feed_backward(NVNET *nnet);

int nvnet_update_params(NVNET *nnet, double rate);
//int nvnet_mmtupdate_params(NVNET *nnet, double rate);
int nvnet_mmtupdate_params(NVNET *nnet, double rate, double mfrict);

int nvnet_buff_params(NVNET *nnet);
int nvnet_restore_params(NVNET *nnet);
int nvnet_check_gradient(NVNET *nnet, const double *tv,
                        double (*loss_func)(double, const double, int) );
void free_nvnet(NVNET *nnet);

/* set param */
//void  nnc_set_param(double learn_rate);
void  nnc_set_param(double learn_rate, double dmfric);
void nnc_set_learnrate(double learn_rate);
void nnc_set_mfrict(double mfric);
double random_btwone(void);

/* print params */
void nvcell_print_params(const NVCELL *nvcell);
void nvlayer_print_params(const NVLAYER *layer);
void nvnet_print_params(const NVNET *nnet);


/* loss func */
double func_lossMSE(double out, const double tv, int token);
double func_lossCrossEntropy(double out, double tv, int token);
//double func_lossCrossEntropy(double *out, double *tv, int classes, int token);

/* others */
bool  gradient_isclose(double da, double db);

/* NVLAYER transfer/activation functions */
int func_softmax(NVLAYER *layer, int token);

#endif
