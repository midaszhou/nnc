/*----------------------------------------------------------------------
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License version 2 as
published by the Free Software Foundation.

Midas Zhou
-----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
//#include <sys/time.h>
#include <string.h>
#include "nnc.h"


#define ERR_LIMIT	0.00001

int main(void)
{

	int i,j;
	int count=0;

	int wh_cells=3; /* wh layer cell number */
	int wo_cells=1; /* wo layer cell number */
	int wh_inputs=3; /* number of input data for each input/hidden nvcell */
	int wo_inputs=3; /* number of input data for each output nvcell */

	double err;
	int ns=8; /* input sample number + teacher value */

	double pin[8*4]= /* 3 input + 1 teacher value */
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


	double data_input[3];


while(1)  {  /* test while */


/*  <<<<<<<<<<<<<<<<<  Create Neuron Net >>>>>>>>>>>>>  */
	/* 1. creat an input template nvcell */
	NVCELL *wh_tempcell=new_nvcell(wh_inputs,NULL,data_input,NULL,0,func_sigmoid); /* input cell */
	nvcell_rand_dwv(wh_tempcell);
	/* create wh(input) layer */
        NVLAYER *wh_layer=new_nvlayer(wh_cells,wh_tempcell);

	/* 2. creat an output template nvcell */
	NVCELL *wo_tempcell=new_nvcell(wo_inputs, wh_layer->nvcells, NULL,NULL,0,func_sigmoid); /* input cell */
	nvcell_rand_dwv(wo_tempcell);
	/* create wo(output) layer */
        NVLAYER *wo_layer=new_nvlayer(wo_cells,wo_tempcell);


/*  <<<<<<<<<<<<<<<<<  NNC Learning Process  >>>>>>>>>>>>>  */
	nnc_set_param(5.0); /* set learn rate */
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

			/* 2. feed forward wh,wo layer */
			nvlayer_feed_forward(wh_layer);
			nvlayer_feed_forward(wo_layer);

			/* 3. get err sum up */
			err += (wo_layer->nvcells[0]->dout - pin[3+i*4])
				 * (wo_layer->nvcells[0]->dout - pin[3+i*4]);

			/* 4. feed backward wo,wh layer, and update model params */
			nvlayer_feed_backward(wo_layer,pin+(3+i*4));
		        nvlayer_feed_backward(wh_layer,NULL);
		}
		count++;

		if( (count&255) == 0)
			printf("	%dth learning, err=%0.8f \n",count, err);
  	}
	printf("	%dth learning, err=%0.8f \n",count, err);
	printf("Finish %d times batch learning!. \n\n",count);


/*  <<<<<<<<<<<<<<<<<  Test Learned NN Model  >>>>>>>>>>>>>  */
	printf("----------- Test learned NN Model -----------\n");
	for(i=0;i<ns;i++)
    	{
		/* update data_input */
		memcpy(data_input, pin+4*i,wh_inputs*sizeof(double));

		/* feed forward wh,wo layer */
		nvlayer_feed_forward(wh_layer);
		nvlayer_feed_forward(wo_layer);

		/* print result */
		printf("Input: ");
		for(j=0;j<wh_inputs;j++)
			printf("%lf ",data_input[j]);
		printf("\n");
		printf("output: %lf \n",wo_layer->nvcells[0]->dout);
	}


/*  <<<<<<<<<<<<<<<<<  Destroy NN  >>>>>>>>>>>>>  */

	free_nvcell(wh_tempcell);
	free_nvcell(wo_tempcell);

	free_nvlayer(wh_layer);
	free_nvlayer(wo_layer);

	usleep(100000);

} /* end test while */

	return 0;
}


