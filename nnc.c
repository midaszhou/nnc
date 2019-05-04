/*----------------------------------------------------------------------
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License version 2 as
published by the Free Software Foundation.

Glossary/Concept:

1. transfer/output/activation function
   devtransfer:derivative function of transfer function
2. loss/error = y(nn resutl) - t(real result)
3. gradient function
4. feedbackward/backpropagation
5. batch-learn and single-learn

dlrate:	learning rate


TODO:
1. Store and deploy model.

Note:
1. a single neural cell with N inputs:
	1. Input:	 x1,x2,x3,.....xn
	2. Weigths:	 w1,w2,w3,.....wn
	3. Bias:	 bias value
	4. sum up:	 u=x1*w1+x2*w2+x3*w3+....+xn*wn -bias

	5. transfer function:   f(u)

	6. activation output:	Z=f(u)

2. Sometimes the err increases monotonously and NERVER converge, it's
   necessary to observe such condition and reset weights and bias
   for all cells.


Midas Zhou
midaszhou@yahoo.com
-----------------------------------------------------------------------*/
#include "nnc.h"
#include "actfs.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>


static double dlrate=20.0;       /* learning rate for all nvcells */


/*--------------------------------------
 * set parameters for NNC
--------------------------------------*/
void  nnc_set_param(double learn_rate)
{
	dlrate=learn_rate;
}


///////////////////////////     Nerve Cell/Layer/Net Concept     ///////////////////////

/*-----------------------------------------------------------------------
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
-------------------------------------------------------------------------*/
NVCELL * new_nvcell( unsigned int nin, NVCELL * const *incells,
				double *din, double *dw, double bias, double (*transfer)(double, double,int ) )
{
	int i;

	/* check input param */
	if(nin==0 ) {
		printf("Init a new NCELL: input param 'nin' error.\n");
		return NULL;
	}

	/* calloc ncell->din and dw */
	NVCELL *ncell=calloc(1,sizeof(NVCELL));
	if(ncell==NULL) {
		printf("Init a new NCELL: fail to calloc ncell.\n");
		return NULL;
	}
	ncell->dw=calloc(nin,sizeof(double));
	if(ncell->dw==NULL) {
		printf("Init a new NCELL: fail to calloc dw.\n");
		free(ncell);
		return NULL;
	}

	/* assign din and dw, or leave it for later assignment  */
	if(din !=NULL) {
		ncell->din=din; /* pass only a pointer to din */
	}
	if(dw !=NULL) {
		for(i=0;i<nin;i++) {
			ncell->dw[i]=dw[i];
			printf("dw[%d]=%f \n",i, ncell->dw[i]);
		}
	}

	/* assign incells, valve and tranfer function */
	ncell->nin=nin;
	ncell->incells=incells;
	ncell->dv=bias;
	ncell->dsum=0.0;
	ncell->transfunc=transfer;

	return ncell;
}

/*--------------------------------------
 * Params:
 * 	@ncell	pointer to a nerve cell.
--------------------------------------*/
void free_nvcell(NVCELL *ncell)
{
	if(ncell==NULL) return;

	if(ncell->dw != NULL)
		free(ncell->dw);

	free(ncell);

	ncell=NULL;
}

/*--------------------------------------
 * Initialize dw and dv for a nvcell
 *
--------------------------------------*/
int nvcell_rand_dwv(NVCELL *ncell)
{
	int i;

	if(ncell==NULL || ncell->dw==NULL)
		return -1;

	/* random dw */
	for(i=0; i < ncell->nin; i++)
		ncell->dw[i]=random_btwone();

	/* random dv */
	ncell->dv=random_btwone();

	return 0;
}



/*-----------------------------------------------
 * Note:
 *	1. Create a new nerve layer with nc nvcells inside.
 *	2. All nvcells of the nvlayer are copied from the template_cell
 *	   so they have same param set inside.
 * Params:
 * 	@nc	number of nerve cells in the layer;
 *	@template_cell	a template cell for layer->nvcells.
 * Return:
 *	pointer to NVLAYER	OK
 *	NULL			fails
--------------------------------------------------*/
NVLAYER *new_nvlayer(int nc, const NVCELL *template_cell)
{
	int i,j;

	/* check input param */
	if(nc==0 ) {
		printf("Init a new NVLAYER: input param 'nc' error.\n");
		return NULL;
	}

	if( template_cell ==NULL ) {
		printf("Init a new NVLAYER: template cell is NULL.\n");
		return NULL;
	}

	/* calloc NVLAYER */
	NVLAYER *layer=calloc(1,sizeof(NVLAYER));
	if(layer==NULL)
		return NULL;

	/* calloc NVCELL* */
	layer->nvcells=calloc(nc, sizeof(NVCELL *));
	if(layer->nvcells==NULL) {
		free(layer);
		return NULL;
	}

	/* assign nc */
	layer->nc=nc;

	/* create nvcells as per template_cell */
	for(i=0;i<nc;i++) {
		layer->nvcells[i]=new_nvcell( template_cell->nin, template_cell->incells,
					      template_cell->din,template_cell->dw, template_cell->dv,
					      template_cell->transfunc
					);
		/* if fail, release all */
		if(layer->nvcells[i]==NULL) {
				printf("Init a new NVLAYER: fail to create new_nvcell!.\n");
				for(j=0;j<i;j++)
					free_nvcell(layer->nvcells[j]);
				free(layer->nvcells);
				free(layer);

				return NULL;
		}
	}

	return layer;
}


/*--------------------------------------
 * Params:
 * 	@layer	pointer to a nerve layer.
---------------------------------------*/
void free_nvlayer(NVLAYER *layer)
{
	int i;

	if(layer==NULL)
		return;

	if(layer->nvcells==NULL) {
		free(layer);
		layer=NULL;
	}

	/* else if nvcells is not NULL */
	for(i=0; i < layer->nc; i++) {
		if(layer->nvcells[i] != NULL)
			free_nvcell(layer->nvcells[i]);
	}

	free(layer);
	layer=NULL;
}


///////////////////////////     Nerve Cell/Layer/Net Functions     ///////////////////////


/*----------------------------------------------
 * Note:
 *	A feed forward function for a nerve cell.
 * Params:
 * 	@nvcell		a nerve cell;
 * Return:
 *		0	OK
 *		<0	fails
-----------------------------------------------*/
int nvcell_feed_forward(NVCELL *nvcell)
{
	int i;

	/* check input param */
	if(nvcell==NULL || nvcell->transfunc==NULL)
			return -1;

	/* 1. reset dsum and derr every time before feedforward */
	nvcell->dsum=0;
	nvcell->derr=0;

	/* 2. Calculate sum of Xn*Wn */
	/* 2.1 Get input from a data array */
	if(nvcell->din != NULL) {
		for(i=0; i < nvcell->nin; i++) {
			nvcell->dsum += (nvcell->din[i]) * (nvcell->dw[i]);
		}
		/* applay dv */
		nvcell->dsum -= nvcell->dv;
	}
	/* 2.2 OR, get data from ahead nvcells' output */
	else if( nvcell->incells !=NULL ) {
		/* check data */
		if(nvcell->incells[0]==NULL) {
			printf("%s: nvcell->incells[x] unavailable! \n",__func__);
			return -2;
		}
		for(i=0; i < nvcell->nin; i++) {
			nvcell->dsum += (nvcell->incells[i]->dout) * (nvcell->dw[i]);
		}
		/* applay dv */
		nvcell->dsum -= nvcell->dv;
	}
	/* 2.3 OR, input data unavailable ! */
	else {
		printf("%s: input data unavailable!\n",__func__);
		return -3;
	}
//	printf(" dsum=%f, dv=%f \n",nvcell->dsum, nvcell->dv);

	/* 3. Calculate output with transfer function */
//	nvcell->dout=(*nvcell->transfunc)(nvcell->dsum - nvcell->dv, NORMAL_FUNC);
	nvcell->dout=(*nvcell->transfunc)(nvcell->dsum, 0, NORMAL_FUNC);  /* f=0 */

	return 0;
}



/*------------------------------------------------------
 * Note:
 *	1. A feed_backward/backpropagation function for a nerve cell.
 *	2. !!!Assume that the nvcell is well confiured and prepared,
 *	   and with result of last feed_forward calculation.
 * Params:
 * 	@nvcell		a nerve cell;
 *	@tv		a teacher's value of output nvcells;
 *			NULL for non_output nvcells
 * Return:
 *		0	OK
 *		<0	fails
--------------------------------------------------------*/
int nvcell_feed_backward(NVCELL *nvcell, const double *tv)
{
	int i;

	/* check input param */
	if(nvcell==NULL || nvcell->transfunc==NULL)
			return -1;

        /* 1. calculate error/loss */
	/* if it's output nvcell */
	if(tv !=NULL )
 	{
  /* ---- LOSS FUNCTION for output nvcell : loss=(t-y)*f'(y) */
		nvcell->derr=(*tv - nvcell->dout) * (nvcell->transfunc(nvcell->dsum, nvcell->dout, DERIVATIVE_FUNC));
	}
	/* else if it's non_output nvcell */
	else
	{
  /* ---- LOSS FUNCTION for NON_output nvcell : loss=derr*f'(y) */
		/* assume that derr has already been feeded backed from the next layer */
		nvcell->derr= nvcell->derr * (nvcell->transfunc(nvcell->dsum, nvcell->dout,DERIVATIVE_FUNC));
	}

//	printf("%s,--- update dw and v --- \n",__func__);

	/* 2. update weight by learning */
	/* 2.1 If NOT input nvcells,  din is ahead nvcell's dout */
	if( nvcell->incells !=NULL && nvcell->incells[0] != NULL) {
		/* update dw */
		for(i=0; i< nvcell->nin; i++) {
			nvcell->dw[i] += dlrate*(nvcell->incells[i]->dout)*(nvcell->derr); /* already put previout output in din[] */

		/* 3. feed back loss to previous nvcell */
   /* ---- LOSS BACKP_ROPAGATION FUNCTION : incell[x]_derr += dw[x]*derr, sum of all next layer feedback error */
			nvcell->incells[i]->derr += (nvcell->dw[i])*(nvcell->derr);
		}
	}

	/* 2.2 ELSE: assume its an input nvcell, get din directly. */
	else if(nvcell->din !=NULL) {
		/* update dw */
		for(i=0; i< nvcell->nin; i++) {
			nvcell->dw[i] += dlrate*(nvcell->din[i])*(nvcell->derr); /* already put previout output in din[] */
		}
	}

	/* 2.3 Data error */
	else {
		printf("%s: nvcell->incells[x] and din[x] invalid!\n",__func__);
		return -2;
	}

	/* 4. update bias value by learning */
	nvcell->dv += dlrate*(-1.0)*(nvcell->derr); /* bias deemed as a special kind of weight, with din[x]=-1 */

	return 0;
}



/*----------------------------------------------
 * Note:
 *	A feed forward function for a nerve layer.
 * Params:
 * 	@layer	a nerve layer;
 * Return:
 *		0	OK
 *		<0	fails
-----------------------------------------------*/
int nvlayer_feed_forward(NVLAYER *layer)
{
	int i;
	int ret;

	/* check layer */
	if(layer==NULL || layer->nvcells==NULL)
		return -1;

	/* feed forward all nvcells in the layer */
	for(i=0; i< layer->nc; i++) {
		ret=nvcell_feed_forward(layer->nvcells[i]);
		if(ret !=0) return ret;
	}

	return 0;
}


/*----------------------------------------------
 * Note:
 *	A feed forward function for a nerve layer.
 * Params:
 * 	@layer	a nerve layer;
 *	@tv		a teacher's value of output nvcells;
 *			NULL for non_output nvcells
 * Return:
 *		0	OK
 *		<0	fails
-----------------------------------------------*/
int nvlayer_feed_backward(NVLAYER *layer, double *tv)
{
	int i;
	int ret;

	/* check layer */
	if(layer==NULL || layer->nvcells==NULL)
		return -1;

	/* feed forward all nvcells in the layer */
	for(i=0; i< layer->nc; i++) {
		ret=nvcell_feed_backward(layer->nvcells[i],tv);
		if(ret !=0) return ret;
	}

	return 0;
}


/*----------------------------------------------
 * A feed forward function for a nerve NET.
 * Params:
 * 	@nvcell		a nerve cell;
 * Return:
 *		0	OK
 *		<0	fails
-----------------------------------------------*/
int nvnet_feed_forward(NVNET *nnet)
{





}



/*------------------------------------------------
 * Params:
 * 	@nnet	a well prepared/confiured nerve net;
 *		with input data in din.
 *	@tv	teacher's value
 * Return:
 *		0	OK
 *		<0	fails
------------------------------------------------*/
int nvcell_get_loss(NVNET *nnet, double tv)
{


}


/*------------------------------------------------------
 * Pass a pointer of data array to nvcell->din.
 * Params:
 * 	@cell		a nerve cell;
 *	@data		array of input data for cell.din
 * Return:
 *		0	OK
 *		<0	fails
-------------------------------------------------------*/
int  nvcell_input_data(NVCELL *cell, double *data)
{
	/* check input */
	if( cell==NULL || data==NULL )
		return -1;

//	if( cell->din !=NULL) {
//		printf("%s: WARNING!  nvcell->din is NOT empty!\n",__func__);
//	}

	/* set input data */
	cell->din = data;

	return 0;
}


///////////////////////////    Common Math     ///////////////////////

/*----------------------------------------------------
 * Generate a random double between -1 to 1 for dw[]
----------------------------------------------------*/
double random_btwone(void)
{
	double rnd;
        struct timeval tmval;

	/* rand seeds */
        gettimeofday(&tmval,NULL);
        srand(tmval.tv_usec);

	while( (rnd=(double)rand()/RAND_MAX) == 1.0 );
	rnd=rnd*2-1;

	return rnd;
}
