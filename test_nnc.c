/*----------------------------------------------------------------------
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License version 2 as
published by the Free Software Foundation.


Some hints:
1. As number of layers increases, learn rate shall be smaller.
2. Simple minds learn simple things.



Midas Zhou
midaszhou@yahoo.com
-----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
//#include <sys/time.h>
#include <string.h>
#include "nnc.h"
#include "actfs.h"

#define ERR_LIMIT	0.0001

int main(void)
{

	int i,j;
	int count=0;

	int wi_cellnum=3; /* wi layer cell number */
	int wm_cellnum=3; /* wm layer cell number */
	int wo_cellnum=1; /* wo layer cell number */

	int wi_inpnum=3; /* number of input data for each input/hidden nvcell */
	int wm_inpnum=3; /* number of input data for each middle nvcell */
	int wo_inpnum=3; /* number of input data for each output nvcell */

	double err;
	int ns=8; /* input sample number + teacher value */

	double pin[8*4]= /* 3 input + 1 teacher value */
#if 0 /* when [0]+[1]+[2]>=2, output [3]=1 */
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
#endif

#if 1 /* Logic: [3]=(-1)^([0]+[1]+[2]) */
{
1,1,1,-1,
1,1,0,1,
1,0,1,1,
1,0,0,-1,
0,1,1,1,
0,1,0,-1,
0,0,1,-1,
0,0,0,1,
};
#endif


#if 1 /* ----- test NEW and FREE ---- */

NVCELL *tmpcell=NULL;
NVLAYER *tmplayer=NULL;
NVNET *nnet=NULL;

int k=0;
while(1) {
	k++;

	tmpcell=new_nvcell(100, NULL, NULL, NULL, 0, func_TanSigmoid);
        tmplayer=new_nvlayer(10,tmpcell);

	free_nvcell(tmpcell);
	free_nvlayer(tmplayer);

	nnet=new_nvnet(10000);
	free_nvnet(nnet);
//	printf(" ------ %d ------\n",k);
	usleep(10000);
}
#endif


	double data_input[3];


//while(1)  {  /* test while */


/*  <<<<<<<<<<<<<<<<<  Create Neuron Net >>>>>>>>>>>>>  */
	/* 1. creat an input nvlayer */
	NVCELL *wi_tempcell=new_nvcell(wi_inpnum,NULL,data_input,NULL,0,func_TanSigmoid);//PReLU);//func_TanSigmoid); /* input cell */
	nvcell_rand_dwv(wi_tempcell);
        NVLAYER *wi_layer=new_nvlayer(wi_cellnum,wi_tempcell);

	/* 2. creat a mid nvlayer */
	NVCELL *wm_tempcell=new_nvcell(wm_inpnum,wi_layer->nvcells,NULL,NULL,0,func_TanSigmoid);//sigmoid); /* input cell */
	nvcell_rand_dwv(wm_tempcell);
        NVLAYER *wm_layer=new_nvlayer(wm_cellnum,wm_tempcell);

	/* 3. creat an output nvlayer */
	NVCELL *wo_tempcell=new_nvcell(wo_inpnum, wm_layer->nvcells, NULL,NULL,0,func_TanSigmoid);//ReLU);//sigmoid); /* input cell */
//	NVCELL *wo_tempcell=new_nvcell(wo_inpnum, wi_layer->nvcells, NULL,NULL,0,func_PReLU);//func_sigmoid); /* input cell */
	nvcell_rand_dwv(wo_tempcell);
        NVLAYER *wo_layer=new_nvlayer(wo_cellnum,wo_tempcell);


/*  <<<<<<<<<<<<<<<<<  NNC Learning Process  >>>>>>>>>>>>>  */
	nnc_set_param(0.015); /* set learn rate */
	err=10; /* give an init value to trigger while() */

  	printf("NN model starts learning ...\n");
  	while(err>ERR_LIMIT)
  	{
		/* reset err */
		err=0.0;

		/* batch learning */
		for(i=0;i<ns;i++)
    		{
			/* 1. update data_input */
			memcpy(data_input, pin+4*i,3*sizeof(double));

			/* 2. feed forward wi->wm->wo layer */
			nvlayer_feed_forward(wi_layer);
			nvlayer_feed_forward(wm_layer);
			nvlayer_feed_forward(wo_layer);

			/* 3. get err sum up */
//			err += (wo_layer->nvcells[0]->dout - pin[3+i*4])
//				 * (wo_layer->nvcells[0]->dout - pin[3+i*4]);

			err += nvlayer_mean_loss(wo_layer, pin+(3+i*4), func_lossMSE);


			/* 4. feed backward wo->wm->wi, and update model params */
			nvlayer_feed_backward(wo_layer); //pin+(3+i*4));
		        nvlayer_feed_backward(wm_layer);
		        nvlayer_feed_backward(wi_layer);

		}
		count++;

		if( (count&255) == 0)
			printf("	%dth learning, err=%0.8f \n",count, err);
  	}

	printf("	%dth learning, err=%0.8f \n",count, err);
	printf("Finish %d times batch learning!. \n",count);

	printf("\n----------- Print Model Params -----------");
	printf("\n		[ wi_layer ]\n");
	nvlayer_print_params(wi_layer);
	printf("\n		[ wm_layer ]\n");
	nvlayer_print_params(wm_layer);
	printf("\n		[ wo_layer ]\n");
	nvlayer_print_params(wo_layer);

/*  <<<<<<<<<<<<<<<<<  Test Learned NN Model  >>>>>>>>>>>>>  */
	printf("\n----------- Test learned NN Model -----------\n");
	for(i=0;i<ns;i++)
    	{
		/* update data_input */
		memcpy(data_input, pin+4*i,wi_inpnum*sizeof(double));

		/* feed forward wi->wm->wo layer */
		nvlayer_feed_forward(wi_layer);
		nvlayer_feed_forward(wm_layer);
		nvlayer_feed_forward(wo_layer);

		/* print result */
		printf("Input: ");
		for(j=0;j<wi_inpnum;j++)
			printf("%lf ",data_input[j]);
		printf("\n");
		printf("output: %lf \n",wo_layer->nvcells[0]->dout);
	}

/*  <<<<<<<<<<<<<<<<<  Destroy NN Model >>>>>>>>>>>>>  */

	free_nvcell(wi_tempcell);
	free_nvcell(wm_tempcell);
	free_nvcell(wo_tempcell);

	free_nvlayer(wi_layer);
	free_nvlayer(wm_layer);
	free_nvlayer(wo_layer);

	usleep(100000);

//} /* end test while */

	return 0;
}


