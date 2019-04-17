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

#define DERIVATIVE_FUNC	1  /* to switch to derivative calculation in a function */
#define NORMAL_FUNC	0

double dlrate=10.0;	/* learning rate for all nvcells */

typedef struct nerve_cell NVCELL; 	/* neuron, or nerve cell */
/*
 * 1. If incells==NULL, then it's an input cell.
 *
 */
struct nerve_cell
{
	int nin; 	/* number of dendrite receivers */
	NVCELL * const *incells; /* array of input cells, whose outputs are inputs for this cell
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


/* Function declaration */
NVCELL * new_nvcell( unsigned int nin, const NVCELL **incells,
				double *din, double *dw, double bias, double (*transfer)(double, int ) );
void free_nvcell(NVCELL *ncell);
int nvcell_rand_dwv(NVCELL *ncell);
int nvcell_feed_forward(NVCELL *nvcell);
int nvcell_feed_backward(NVCELL *nvcell, const double *tv);
int nvcell_input_data(NVCELL *cell, double *data);


NVLAYER *new_nvlayer(int nc, NVCELL **cells);
void free_nvlayer(NVLAYER *layer);


double func_step(double u, int token);
double func_sigmoid(double u, int token);

double random_btwone(void);


/*-----------------------------
	MAIN FUNCTION
-----------------------------*/
int main(void)
{
	#define ERR_LIMIT	0.001

	int i,j;
	int count=0;

	int num_incells=3;
	int num_outcells=1;

	int wh_inputs=3; /* number of input data for each input/hidden nvcell */
	int wo_inputs=3; /* number of input data for each output nvcell */

	double err;
int ns=8; /* input sample number + teacher value */
double pin[8][4]= /* 3 input + 1 teacher value */
{
1,1,1,1,
1,1,0,1,
1,0,1,1,
1,0,0,0,
0,1,1,1,
0,1,0,0,
0,0,1,0,
0,0,0,0,
};



//while(1)  {  /* test while */

	/* INPUT CELLS */
	NVCELL *ncell_wh1=new_nvcell(wh_inputs,NULL,NULL,NULL,0,func_sigmoid); /* input cell */
	NVCELL *ncell_wh2=new_nvcell(wh_inputs,NULL,NULL,NULL,0,func_sigmoid); /* input cell */
	NVCELL *ncell_wh3=new_nvcell(wh_inputs,NULL,NULL,NULL,0,func_sigmoid); /* input cell */
	/* init dw dv */
	nvcell_rand_dwv(ncell_wh1);
	nvcell_rand_dwv(ncell_wh2);
	nvcell_rand_dwv(ncell_wh3);

	/* OUTPUT CELLS */
	NVCELL **incells=calloc(num_incells,sizeof(NVCELL *));
	incells[0]=ncell_wh1;
	incells[1]=ncell_wh2;
	incells[2]=ncell_wh3;
	NVCELL *ncell_wo=new_nvcell(wo_inputs,incells,NULL,NULL,0,func_sigmoid); /* output cell */
	/* init dw dv */
	nvcell_rand_dwv(ncell_wo);




  err=10.0;
  while(err>ERR_LIMIT)
  {
	err=0.0;
	/* test data learning */
	for(i=0;i<ns;i++)
    	{

		/* feed data to input_cells directly  */
		nvcell_input_data(ncell_wh1, &pin[0][0]+(wh_inputs+1)*i); /* +1 as teacher value */
		nvcell_input_data(ncell_wh2, &pin[0][0]+(wh_inputs+1)*i);
		nvcell_input_data(ncell_wh3, &pin[0][0]+(wh_inputs+1)*i);

		/* feed forward: hidden layer */
		nvcell_feed_forward(ncell_wh1);
		nvcell_feed_forward(ncell_wh2);
		nvcell_feed_forward(ncell_wh3);

		/* feed forward: output layer */
		nvcell_feed_forward(ncell_wo);
		printf("nnc output: %f \n",ncell_wo->dout);

		/* err sumup */
		err += (ncell_wo->dout-pin[i][3])*(ncell_wo->dout-pin[i][3]);

		/* dw dv learning */
		nvcell_feed_backward(ncell_wo,&pin[0][0]+(wh_inputs+1)*i+3);
		nvcell_feed_backward(ncell_wh1,NULL);
		nvcell_feed_backward(ncell_wh2,NULL);
		nvcell_feed_backward(ncell_wh3,NULL);
	}
	count++;

	printf("-------- %dth learning, err=%lf -------\n",count, err);

  }

	/* test learned NNC model */
	for(i=0;i<ns;i++)
    	{

		printf("------------- after learning -------------\n");
		/* feed data to input_cells directly  */
		nvcell_input_data(ncell_wh1, &pin[0][0]+(wh_inputs+1)*i); /* +1 as teacher value */
		nvcell_input_data(ncell_wh2, &pin[0][0]+(wh_inputs+1)*i);
		nvcell_input_data(ncell_wh3, &pin[0][0]+(wh_inputs+1)*i);

		/* feed forward: hidden layer */
		nvcell_feed_forward(ncell_wh1);
		nvcell_feed_forward(ncell_wh2);
		nvcell_feed_forward(ncell_wh3);

		/* feed forward: output layer */
		nvcell_feed_forward(ncell_wo);

		/* print result */
		printf("Input: ");
		for(j=0;j<wh_inputs;j++) {
			printf("%lf, ",pin[i][j]);
		}
		printf("\n");
		printf("output: %lf \n",ncell_wo->dout);
	}


	/* free all */
	free_nvcell(ncell_wh1);
	free_nvcell(ncell_wh2);
	free_nvcell(ncell_wh3);
	free_nvcell(ncell_wo);
	free(incells);

	usleep(50000);

//} /* end test while */


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

	/* assign din and dw, or leave it for later assignment  */
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

/*--------------------------------------
 * Initialize dw and dv for a nvcell
 *
--------------------------------------*/
int nvcell_rand_dwv(NVCELL *ncell)
{
	int i;

	if(ncell==NULL || ncell->dw==NULL)
		return -1;

	/* random dw and dv */
	for(i=0; i < ncell->nin; i++)
		ncell->dw[i]=random_btwone();
	ncell->dv=random_btwone();

	return 0;
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
   if(token==NORMAL_FUNC) {
	if(u>=0)
		return 1.0;
	else
		return 0.0;
   }
   /* Derivative func */
   else {
	printf("%s: Derivative function NOT defined!!! \n",__func__);
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
   if(token==NORMAL_FUNC) {
	return 1.0/(1.0+exp(-u));
   }
   /* if DERIVATIVE_FUNC Derivative func*/
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
//			printf("din from incells: din[%d]=%f \n",i,nvcell->din[i]);
		}
	}
	/* 1. ELSE:  use nvcell->din[] as input data */

	/* 2. Calculate sum of Xn*Wn */
	nvcell->dsum=0; /* reset dsum */
	for(i=0; i < nvcell->nin; i++) {
		nvcell->dsum += (nvcell->din[i]) * (nvcell->dw[i]);
	}
//	printf(" dsum=%f, dv=%f \n",nvcell->dsum, nvcell->dv);

	/* 3. Calculate output with transfer function */
	nvcell->dout=(*nvcell->transfunc)(nvcell->dsum - nvcell->dv, NORMAL_FUNC);

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
	/* if output nvcell */
	if(tv !=NULL )
 	{
     /* ---- LOSS FUNCTION for output nvcell : loss=(t-y)*f'(y) */
		nvcell->derr=(*tv - nvcell->dout) * (nvcell->transfunc(nvcell->dout,DERIVATIVE_FUNC));
	}
	/* else if non_output nvcell */
	else
	{
     /* ---- LOSS FUNCTION for NON_output nvcell : loss=derr*f'(y) */
		/* assume that derr has already been feeded backed from the next layer */
		nvcell->derr= nvcell->derr * (nvcell->transfunc(nvcell->dout,DERIVATIVE_FUNC));
	}

	/* 2. update weight by learning */
	//printf("%s,--- update dw and v --- \n",__func__);
	for(i=0; i< nvcell->nin; i++) {
		nvcell->dw[i] += dlrate*(nvcell->din[i])*(nvcell->derr); /* already put previout output in din[] */

		/* 3. feed back loss to previous nvcell, if it's NOT input nvcells! */
		if( nvcell->incells !=NULL && nvcell->incells[i] != NULL) {

     /* ---- LOSS BACKP_ROPAGATION FUNCTION : incell[x]_derr=dw[x]*derr */
			(nvcell->incells[i])->derr = (nvcell->dw[i])*(nvcell->derr);
		}
	}

	/* 4. update bias value by learning */
	nvcell->dv += dlrate*(-1.0)*(nvcell->derr); /* bias deemed as a special kind of weight, with din[x]=-1 */

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
