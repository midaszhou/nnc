/*----------------------------------------------------------------------
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License version 2 as
published by the Free Software Foundation.

Neural Network Description:
   Input(28*28) |<<<    Input_Layer(28*28i,20o)    >>>|<<<   Hidden_Layer(20i,20o)  >>>|<<<  Output_Layer(20i,10o)  >>>|

Reference:
1. Neural Network Training Model:
   https://peterroelants.github.io/posts/neural-network-implementation-part05/

2. MNIST handwrriten digit database:
   http://yann.lecun.com/exdb/mnist/


Journal:
2023-07-26: Create the file.

Midas Zhou
知之者不如好之者好之者不如乐之者
-----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>

#include "nnc.h"
#include "actfs.h"


#define ERR_LIMIT       0.0025 //0.001
#define BUFFER_IMGDATA  0  /* 1---MNIST image data are buffered to train_imgdata/test_imgdata
				  Except MNIST label data, they are ALWAYS buffered to train_target[]/test_target[]
			      0---NO buffer, read imgdata from mmap
			    */

int main(void)
{
	int i,j,k;
        int count=0;
        int loop=0;

        /* NVCELL Layer nodes */
        int wi_inpnum=28*28; /* number of input data for each input/hidden nvcell */
        int wi_cellnum=20; /* wi layer cell number */

        int wm_inpnum=wi_cellnum; /* number of input data for each middle nvcell */
        int wm_cellnum=20; /* wm layer cell number */

        int wo_inpnum=wm_cellnum; /* number of input data for each output nvcell */
        int wo_cellnum=10; /* wo layer cell number */


	/* Train data buff */
	const int TRAIN_IMGTOTAL=10000; //50000;
	/***
	 err_limit=0.001, train_imgtotal=5000, accuracy ~85%
	 err_limit=0.001, train_imgtotal=20000, accuracy ~91%
	 err_limit=0.001,mean_err=0.00381249, train_imgtotal=50000, accuracy ~94%
         */

	/* Note: each 28x28 pixles, gray values[0 255] are normalized to [0 1.0] */
	double *train_imgdata = NULL;
	if(BUFFER_IMGDATA) {
	 	train_imgdata=(double *)malloc(TRAIN_IMGTOTAL*28*28*sizeof(double));
		if(train_imgdata==NULL) exit(1);
	}
	unsigned char train_target[TRAIN_IMGTOTAL];	   /* digits 0~9 */

	/* Test data buffer */
	const int TEST_IMGTOTAL=5000;
	/* Note: each 28x28 pixles, gray values[0 255] are normalized to [0 1.0] */
	double *test_imgdata=NULL;
	if(BUFFER_IMGDATA) {
		test_imgdata= (double *)malloc(TEST_IMGTOTAL*28*28*sizeof(double));
		if(test_imgdata==NULL) exit(1);
	}
	unsigned char test_target[TEST_IMGTOTAL];	   /* digits 0~9 */


        /* To link it to input_layer->pins */
        double data_input[28*28]; /* Normalized to [0 1.0] */
        double data_target[10]; /* one_hot target values. ONLY one '1' and others are '0'. */

        double err, batch_err, mean_err;
        int nb=1;     /* number of batches */
        int bs=TRAIN_IMGTOTAL/nb; //500; /*  batch size; <=TRAIN_IMGTOTAL  input samples (numbers + teacher value) for each batch training */

        bool gradient_checked=false;


/*  <<<<<<<<<<<<<<<<<  Read data from MNIST database file  >>>>>>>>>>>>>  */

  	////////////* Read MNIST handwriten digit database *////////////

        /***   		---- Image data ---
          [offset]   [type]          [value]             [description]
          0000     32bit integer   0x00000803(2051)     magic number
          0004     32bit integer   10000                number of images
          0008     32bit integer   28                   number of rows
          0012     32bit integer   28                   number of columns
          (followings are all pixel data -----: pdata )
          0016     unsigned byte   ??                   pixel
          0017     unsigned byte   ??                   pixel
          ... ...

            		---- Label data ---
          [offset]   [type]          [value]             [description]
          0000     32bit integer   0x00000801(2049)     magic number
          0004     32bit integer   10000                number of items
          (followings are all label data -----: )
          0008     unsigned byte   ??                   label
          0009     unsigned byte   ??                   label
          ... ...
         */

	const char *train_images_path="train-images.idx3-ubyte";
	const char *train_labels_path="train-labels.idx1-ubyte";

	int fd0, fd1;
	struct stat  sb;
        unsigned int magic;
        unsigned int imgcnt, labelcnt;
        unsigned int  nr,nc;
	void *addr0, *addr1;
        unsigned char *pdata0, *pdata1;
	unsigned char *pTrainImg, *pTestImg;
        //unsigned int offset;      /* offset from pdata */

	/* ------------ Read in data for training ------------ */

        /* M1. Open MNIST image data file */
	fd0=open(train_images_path, O_RDONLY);
        if(fd0<0) {
                printf("%s: Fail to open image data file!\n", __func__);
		exit(1);
        }
	if(fstat(fd0, &sb)<0)
		exit(1);

        /* M2. MMAP image data file */
        //addr0=mmap(NULL, 8*1024*1024, PROT_READ, MAP_PRIVATE, fd0, 0);
        addr0=mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd0, 0);
        if(addr0==MAP_FAILED) {
                        printf("%s: Fail to mmap file!\n", __func__);
			perror("mmap");
			close(fd0);
			exit(1);
         }
	pdata0 = (unsigned char *)addr0;

        /* M3. Read MNIST data header */
        magic=be32toh( *(unsigned int*)(pdata0+0) );
        imgcnt=be32toh( *(unsigned int*)(pdata0+4) );
        nr=be32toh( *(unsigned int*)(pdata0+8) );
        nc=be32toh( *(unsigned int*)(pdata0+12) );
        printf("MNIST '%s': magic=%d, imgcnt=%d, nr=%d, nc=%d\n", train_images_path, magic, imgcnt, nr,nc);
	/* Result:  MNIST t10k-images: magic=2051, imgcnt=10000, nr=28, nc=28 */

	/* M4. Assign pdata to the beginning of pixel data */
        pdata0 +=16; //(unsigned char*)pdata+16;
	pdata0 +=0*28*28; /* Skip some images, they may be errors. */

	pTrainImg = pdata0;
	pTestImg = pdata0+TRAIN_IMGTOTAL*28*28;

if(BUFFER_IMGDATA) {
	/* M5. Copy into train_imgdata[] */
	for(i=0; i<TRAIN_IMGTOTAL*28*28; i++)
		train_imgdata[i] = (double)pdata0[i] / 255.0;

	/* M6. Copy into test_imgdata[] */
	for(i=0; i<TEST_IMGTOTAL*28*28; i++) {
		test_imgdata[i] = (double)pdata0[TRAIN_IMGTOTAL*28*28+i] / 255.0;
		//test_imgdata[i] = (double)pdata[i];
	}
}

        /* MA1. Open MNIST label data file. */
        fd1=open(train_labels_path, O_RDONLY);
        if(fd1<0) {
                printf("%s: Fail to open label data file!\n", __func__);
                exit(1);
        }
	if(fstat(fd1, &sb)<0)
		exit(1);

        /* MA2. MMAP file */
        //addr1=mmap(NULL, 10*1024, PROT_READ, MAP_PRIVATE, fd1, 0);
        addr1=mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd1, 0);
        if(addr1==MAP_FAILED) {
                        printf("%s: Fail to mmap file!\n", __func__);
                        perror("mmap");
                        close(fd1);
                        exit(1);
        }
	pdata1 = (unsigned char *)addr1;

        /* MA3. Read MNIST data header */
        magic=be32toh( *(unsigned int*)(pdata1+0) );
        labelcnt=be32toh( *(unsigned int*)(pdata1+4) );
        printf("MNIST '%s': magic=%d, labelcnt=%d\n", train_labels_path, magic, labelcnt);

	/* MA4. Assign pdata to the beginning of label data */
        pdata1 +=8;
	pdata1 +=0*1; /* Skip some images, they may be errors. */

	/* MA5. Copy into train_target[] */
	for(i=0; i<TRAIN_IMGTOTAL; i++) {
		train_target[i] = pdata1[i];
		//printf("train_target[%d]: %d\n", i, train_target[i]);
	}

	/* MA6. Copy into test_target[] */
	for(i=0; i<TEST_IMGTOTAL; i++) {
		test_target[i] = pdata1[TRAIN_IMGTOTAL+i];
		//printf("test_target[%d]: %d\n", i, train_target[i]);
	}


	/* ------------ TODO:  Read in data for testing ------------ */

  	////////////* END: Read MNIST handwriten digit database *//////////


/*  <<<<<<<<<<<<<<<<<  Create CNN(Convolution Neural Network) >>>>>>>>>>>>>  */
	printf("Create neurial network model...\n");

        /* 1. create an input nvlayer */
        NVCELL *wi_tempcell=new_nvcell(wi_inpnum,NULL, data_input, NULL,0, func_ReLU); //func_TanSigmoid); /* input cell */
        NVLAYER *wi_layer=new_nvlayer(wi_cellnum,wi_tempcell, false);

        /* 2. create a mid nvlayer */
        NVCELL *wm_tempcell=new_nvcell(wm_inpnum,wi_layer->nvcells,NULL,NULL,0, func_ReLU); //sigmoid);
        NVLAYER *wm_layer=new_nvlayer(wm_cellnum,wm_tempcell, false);

        /* 3. create an output nvlayer */
        NVCELL *wo_tempcell=new_nvcell(wo_inpnum, wm_layer->nvcells, NULL,NULL,0, NULL); //NULL); //func_TanSigmoid);//sigmoid); /* input cell */
        NVLAYER *wo_layer=new_nvlayer(wo_cellnum,wo_tempcell, true); /* false , we do not need douts, or no layer transfunc defined */
        wo_layer->transfunc = func_softmax;

        /* 4. Create an nerve net */
        NVNET *nnet=new_nvnet(3); /* 3 layers inside */
        nnet->nvlayers[0]=wi_layer;
        nnet->nvlayers[1]=wm_layer;
        nnet->nvlayers[2]=wo_layer;

        /* 5. Init params */
        nvnet_init_params(nnet);

/*  <<<<<<<<<<<<<<<<<  CNN Training Process  >>>>>>>>>>>>>  */

        /* 6. Set learning_rate and  momentum friction */
        nnc_set_mfrict(0.1);
        /* NOTE:
         * NO MORE USE if call nvnet_update_params()/nvnet_mmtupdate_params() later.
         *  Call nvnet_update_params(nnet, learn_rate), OR  nvnet_mmtupdate_params(nnet, learn_rate, mfrict)
         */

        /* 7. Init err value */
        mean_err=10;

        printf("NN model starts training ...\n");

        /* 8. Batch training */
        count=0;
        gradient_checked=false;
        while( count<10 || (mean_err > ERR_LIMIT && count<3000 ) )
        {
                /* 8.1  Reset batch err */
                batch_err=0.0;

                /* 8.2  batch learning */
                for(nb=0; nb< TRAIN_IMGTOTAL/bs; nb++) {

                    /* Run all samples in the batch */
                    for(i=0; i<bs; i++) {

                        /* 8.2.R1. update ONE sample for input: data_input, data_target */
			#if !BUFFER_IMGDATA /* --- CORSS_CHECK T.1 --- */
                        for(k=0; k<28*28; k++)
				data_input[k]=pTrainImg[(nb*bs+i)*(28*28)+k]/255.0;
			#else /* --- CORSS_CHECK T.1 --- */
			conv_layer->conv3x3->din = train_imgdata+(nb*bs+i)*(28*28);
			#endif
                        for(k=0; k<10; k++)
                                data_target[k] = (k==train_target[nb*bs+i] ? 1.0 : 0.0);

                        /* 8.2.R2. nvnet feed forward, accumlate err values for batch error
                           NOTICE: The output layer has transfunc defined, as func_softmax().
                         */
                        err = nvnet_feed_forward(nnet, data_target, func_lossCrossEntropy); //func_lossMSE);
                        if(isnan(err) || isinf(err) ) { /* If NAN */
                                printf("Return err is nan or inf! Too big learn_rate? or input data unnormalized?\n");
				goto END_WHILE;
                                //exit(1);
                        }
                        batch_err +=err;

                        /* 8.2.R3. nvnet feed backward, update cell->derrs */
                        nvnet_feed_backward(nnet);

                        /* 8.2.R5. update params after feedback(backpropagation) computation */
                        nvnet_update_params(nnet, 0.0025); /* 0.002 learn_rate */
                        /* Note: Big learn_rate(>0.01, example 0.1) will cause all -nan results and quit training! */

                    } /* for(i) */

                } /* for(nb) */

                /* 8.3 Mean err for batch training */
                mean_err = batch_err/(nb*bs);

                /* 8.4 Count training */
                count++;

                printf("Epoch %d: samples=%d, mean_err=%0.8f \n",count, TRAIN_IMGTOTAL, mean_err);

        }

END_WHILE:
        /* 9. Print results */
        printf("        %dth learning, mean_err=%0.8f \n",count, mean_err);
        printf("Finish %d times batch learning!. \n",count);


/*  <<<<<<<<<<<<<<<<<  Test CNN Model  >>>>>>>>>>>>>  */

        int errcnt=0;
        printf("\n----------- Test learned NN Model -----------\n");
        for(i=0; i<TEST_IMGTOTAL; i++)
        {
                /* T1. Update data_input,data_target */
		#if !BUFFER_IMGDATA  /* --- CORSS_CHECK 8.2.R1 --- */
                for(k=0; k<28*28; k++)
			 data_input[k]=pTestImg[i*(28*28)+k]/255.0;
		#else  /* --- CORSS_CHECK 8.2.R1 --- */
		conv_layer->conv3x3->din = test_imgdata+i*(28*28);
		#endif
                for(k=0; k<10; k++)
                         data_target[k] = (k==test_target[i] ? 1.0 : 0.0);

		/* T2. CNN feed forward to predict */
                err = nvnet_feed_forward(nnet, data_target, func_lossCrossEntropy); //func_lossMSE);
                if( isnan(err) || isinf(err) ) { /* If NAN */
                	printf("Return err is nan or inf! Too big learn_rate?\n");
                        exit(1);
                }

                /* T3. Print result */
                printf("[%d] Output: ", i);
                for(k=0; k< wo_layer->nc ; k++)
                        printf("%f, ",wo_layer->douts[k]);
                printf("    ");
                printf("Target: %d ", test_target[i]);
                if(wo_layer->douts[ test_target[i] ] >0.5)
                        printf(" OK\n");
                else {
                        printf(" Fails\n");
                        errcnt++;
                }
        }
	printf("Train samples: %d, Test samples: %d\n", TRAIN_IMGTOTAL, TEST_IMGTOTAL);
        printf("Err/Total: %d/%d   Accuracy: %.2f%%\n", errcnt, TEST_IMGTOTAL, 100.0*(1.0-1.0*errcnt/TEST_IMGTOTAL));


	/* Free data and nvcell/nvnet */
	free(train_imgdata);
	free(test_imgdata);
	free_nvcell(wi_tempcell);
	free_nvcell(wm_tempcell);
	free_nvcell(wo_tempcell);
        free_nvnet(nnet); /* free nvnet also free its nvlayers and nvcells inside */

        /* Unmap and close */
        munmap(addr0, 8*1024*1024);
        close(fd0);
        munmap(addr1, 8*1024);
        close(fd1);


	return 0;
}
