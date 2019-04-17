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


Note:
a single neural cell with N inputs:
	1. Input:	 x1,x2,x3,.....xn
	2. Weigths:	 w1,w2,w3,.....wn
	3. Bias:	 bias value
	4. sum up:	 u=x1*w1+x2*w2+x3*w3+....+xn*wn -bias
	5. transfer function: f(u)
	6. output:	Z=f(u)


Midas Zhou
-----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

double dlrate=10.0;	/* learning rate for all nvcells */

typedef struct nerve_cell NVCELL; 	/* neuron, or nerve cell */
/*
 * 1. If incells==NULL, then it's an input cell.
 *
 */
struct nerve_cell
{
	int nin; 	/* number of dendrite receivers */
	const NVCELL **incells; /* array of input cells, whose outputs are inputs for this cell
				  Only a pointer, will NOT allocate mem space. */
	double *din; 	/* array of input data, mem space to be allocated. */
	double *dw;  	/* array of weights, mem sapce to be allocated */
	double dv;	/* bias value, dsum - dv */
	double dsum;	/* dh, sum of x1*w1+x2*w2+x3*w3+....+xn*wn  ? -dv */
	double (*transfunc)(double,int); /* pointer to a transfer function (also include derivative part) */
//	double (*dev_transfunc)(double); /* pointer to the derivative function of the transfunc */
	double dout;	/* dz, output value, dout=transfunc(dsum-dv) */
	double derr;	/* error(loss) or error(loss) value back feeded(back propagated) from the next layer cell */
};

typedef struct nerve_layer NVLAYER;
struct nerve_layer
{
	int nc;		/* number of NVCELL in the layer */
	NVCELL * *nvcells;	/* array of nerve cells in the layer */
};


typedef struct nerve_net NVNET;
struct nerve_net
{
	int nlayer;		/* number of NVLAYER in the net */
	NVLAYER *nvlayers;       /*  array of nervers for the net */
};


NVCELL * new_nvcell( unsigned int nin, const NVCELL **incells,
				double *din, double *dw, double bias, double (*transfer)(double, int ) );
void free_nvcell(NVCELL *ncell);

NVLAYER *new_nvlayer(int nc, NVCELL **cells);
void free_nvlayer(NVLAYER *layer);

double func_step(double u, int token);
double func_sigmoid(double u, int token);
int nvcell_feed_forward(NVCELL *nvcell);
int nvcell_input_data(NVCELL *cell, double *data);


/*-----------------------------
	MAIN FUNCTION
-----------------------------*/
int main(void)
{
	int i=0;
	double wh1[2]={-2,3};
	double bh1=-1.0;
	double wh2[2]={-2,1};
	double bh2=0.5;
	double wo[2]={-60,94};
	double bo=-1.0;

	double pin[4][2]={ {0,0}, {0,1}, {1,0}, {1,1} };


	NVCELL **incells=calloc(2,sizeof(NVCELL *));

	NVCELL *ncell_wh1=new_nvcell(2,NULL,NULL,wh1,bh1,func_step); /* input cell */
	NVCELL *ncell_wh2=new_nvcell(2,NULL,NULL,wh2,bh2,func_step); /* input cell */

	incells[0]=ncell_wh1;
	incells[1]=ncell_wh2;
	NVCELL *ncell_wo=new_nvcell(2,incells,NULL,wo,bo,func_step); /* output cell */

    for(i=0;i<4;i++)
    {
	/* feed data to input_cells directly  */
	nvcell_input_data(ncell_wh1, &pin[0][0]+2*i);
	nvcell_input_data(ncell_wh2, &pin[0][0]+2*i);
	printf("---------------- i=%d ----------------\n",i);
	printf("pin[0]=%f, pin[1]=%f \n", *(&pin[0][0]+2*i), *(&pin[0][0]+2*i+1) );

	/* feed forward: hidden layer */
	nvcell_feed_forward(ncell_wh1);
	printf("Wh1 result: %f \n",ncell_wh1->dout);
	nvcell_feed_forward(ncell_wh2);
	printf("Wh2 result: %f \n",ncell_wh2->dout);

	/* feed forward: output layer */
	nvcell_feed_forward(ncell_wo);
	printf("wo result: %f \n",ncell_wo->dout);

    }
//	NVLAYER *new_nvlayer(int nc, NVCELL **cells);

	/* free all */
	free_nvcell(ncell_wh1);
	free_nvcell(ncell_wh2);
	free_nvcell(ncell_wo);
	free(incells);

	printf("i=%d,\n",i++);
	usleep(5000);

	return 0;
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
NVCELL * new_nvcell( unsigned int nin, const NVCELL **incells,
				double *din, double *dw, double bias, double (*transfer)(double,int ) )
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
		for(i=0;i<nin;i++) {
			ncell->din[i]=din[i];
			printf("din[%d]=%f \n",i, ncell->din[i]);
		}
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


/*--------------------------------------------
 * Note:
 *	create a new nerve layer.
 * Params:
 * 	@nc	number of nerve cells in the layer;
 *	@cells	array of pointers to nerve cells, if NULL, nerve cells of the layer will be defined later.
 * Return:
 *	pointer to NVCELL	OK
 *	NULL			fails
---------------------------------------------*/
NVLAYER *new_nvlayer(int nc, NVCELL **cells)
{
	int i;

	/* check input param */
	if(nc==0 ) {
		printf("Init a new NVLAYER: input param 'nc' error.\n");
		return NULL;
	}

	/* calloc NVLAYER */
	NVLAYER *layer=calloc(1, sizeof(NVCELL));
	if(layer==NULL)
		return NULL;

	/* calloc NVCELL* */
	layer->nvcells=calloc(nc, sizeof(NVCELL *));
	if(layer->nvcells==NULL) {
		free(layer);
		return NULL;
	}

	/* assign nc,  and nvcells if not NULL */
	layer->nc=nc;
	if(cells != NULL) {
		for(i=0;i<nc;i++) {
			if( cells[i] != NULL )
				layer->nvcells[i]=cells[i];
			else
				printf("Init a new NVLAYER: warning!!! input cells[%d] is NULL!\n",i);
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
			free(layer->nvcells[i]);
	}

	free(layer);
	layer=NULL;
}

///////////////////////////     Nerve Cell/Layer/Net Functions     ///////////////////////

/*-----------------------------------------------
 * Note:
 *	A step transfer function.
 * Params:
 * 	@u	input param for transfer function;
 * Return:
 *		a double.
------------------------------------------------*/
double func_step(double u, int token)
{
   /* Normal func */
   if(token==0) {
	if(u>=0)
		return 1.0;
	else
		return 0.0;
   }
   /* Derivative func */
   else {
	return 0; ///////////////////////////////// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   }
}

/*-----------------------------------------------
 * Note:
 *	A Sigmoid function.
 * Params:
 * 	@u	input param for function;
 *		if token==0:  param for sigmoid f(u)
 *		if token!=0:  f'(u)=u(1-u) !!!!!!
 *	@token	0 ---normal func; 1 ---derivative func
 * Return:
 *		a double.
------------------------------------------------*/
double func_sigmoid(double u, int token)
{
   /* Normal func */
   if(token==0) {
	return 1.0/(1.0+exp(-u));
   }
   /* Derivative func */
   else  {
	return u*(1-u);
  }
}


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

	/* 1. get input data from a nvcell output */
	if(nvcell->incells != NULL) {
		for(i=0; i < nvcell->nin; i++) {
			nvcell->din[i]=nvcell->incells[i]->dout;
			printf("din from incells: din[%d]=%f \n",i,nvcell->din[i]);
		}
	}
	/* 1. ELSE:  use nvcell->din[] as input data */

	/* 2. Calculate sum of Xn*Wn */
	nvcell->dsum=0; /* reset dsum */
	for(i=0; i < nvcell->nin; i++) {
		nvcell->dsum += (nvcell->din[i]) * (nvcell->dw[i]);
	}
	printf(" dsum=%f, dv=%f \n",nvcell->dsum, nvcell->dv);

	/* 3. Calculate output with transfer function */
	nvcell->dout=(*nvcell->transfunc)(nvcell->dsum - nvcell->dv, 0);
}

/*------------------------------------------------------
 * Note:
 *	1. A feed_backward/backpropagation function for a nerve cell.
 *	2. Assume that the nvcell is well confiured and prepared,
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

	/* if output nvcell */
	if(tv !=NULL )
 	{
//		nvcell->derr=(*tv - nvcell->dout) *  /* calculate error */

	}
	/* if non_output nvcell */
	else
	{
		/* assume that derr already feed backed from the next layer */
	}
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
 * 
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
 *	A feed forward function for a nerve cell.
 * Params:
 * 	@cell		a nerve cell;
 *	@data		array of input data for cell.din
 * Return:
 *		0	OK
 *		<0	fails
-------------------------------------------------------*/
int  nvcell_input_data(NVCELL *cell, double *data)
{
	int i;

	/* check input */
	if( cell==NULL || cell->din==NULL || data==NULL )
			return -1;

	/* input data */
	for(i=0; i < cell->nin; i++) {
		cell->din[i] = data[i];
		//printf("din[%d]=%f \n",i,cell->din[i]);
	}

	return 0;
}


