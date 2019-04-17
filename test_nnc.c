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
#include <sys/time.h>
#include "nnc.h"


#define ERR_LIMIT	0.00001

int main(void)
{

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


/*  <<<<<<<<<<<<<<<<<  Create Neuron Net >>>>>>>>>>>>>  */

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



/*  <<<<<<<<<<<<<<<<<  NNC Learning Process  >>>>>>>>>>>>>  */
  nnc_set_param(5.0); /* set learn rate */
  err=10.0;

  printf("NN model starts learning ...\n");
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

		/* err sumup */
		err += (ncell_wo->dout-pin[i][3])*(ncell_wo->dout-pin[i][3]);

		/* dw dv learning */
		nvcell_feed_backward(ncell_wo,&pin[0][0]+(wh_inputs+1)*i+3);
		nvcell_feed_backward(ncell_wh1,NULL);
		nvcell_feed_backward(ncell_wh2,NULL);
		nvcell_feed_backward(ncell_wh3,NULL);
	}
	count++;

	if( (count&255) == 0)
		printf("	%dth learning, err=%0.8f \n",count, err);
  }

  printf("	%dth learning, err=%0.8f \n",count, err);
  printf("Finish!. \n\n");


/*  <<<<<<<<<<<<<<<<<  Test Learned NN Model  >>>>>>>>>>>>>  */
	printf("----------- Test learned NN Model -----------\n");
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

		/* print result */
		printf("Input: ");
		for(j=0;j<wh_inputs;j++) {
			printf("%lf, ",pin[i][j]);
		}
		printf("\n");
		printf("output: %lf \n",ncell_wo->dout);
	}


/*  <<<<<<<<<<<<<<<<<  Destroy NN  >>>>>>>>>>>>>  */
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


