/*----------------------------------------------------------------------
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License version 2 as
published by the Free Software Foundation.

Neural Network Description:
   Input(28*28) |>>> CONV3X3 (26x26x4_out)  >>>|<<< MAXPOOL2X2 (13x13x4_out) >>>|<<< nvCells (10_out) >>>|

Reference:
1. https://victorzhou.com/blog/intro-to-cnns-part-1/
   https://victorzhou.com/blog/intro-to-cnns-part-2/

2. MNIST handwrriten digit database:
   http://yann.lecun.com/exdb/mnist/

Note:
1. Input image data should be normalized to [0 1.0], or loss function
   may return NAN or INF results. So as weights/bias values.
2. Filter number 8 OR 5? Seem 5 is faster and also same accurate.

TODO:
1. Looptest to check mem leakage.

Journal:
2023-07-11: Create the file.

Midas Zhou
知之者不如好之者好之者不如乐之者
-----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <sys/mman.h> /* mmap */
#include <fcntl.h>    /* O_RDONLY */
#include <sys/stat.h>
#include <time.h>

#include "nnc.h"
#include "actfs.h"

#define ERR_LIMIT       0.0025 //0.001
#define BUFFER_IMGDATA  0  /* 1---MNIST image data are buffered to train_imgdata/test_imgdata
				  Except MNIST label data, they are ALWAYS buffered to train_target[]/test_target[]
			      0---NO buffer, read imgdata from MMAP directly
			    */

int main(void)
{
	int i,j,k;
        int count=0;
        int loop=0;

	/* Train data buffer */
	const int TRAIN_IMGTOTAL=10000;
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

	/* Fliters */
	int numFilters=8;

        double err, batch_err, mean_err;
        int nb=1;     /* number of batches */
        int bs=TRAIN_IMGTOTAL/nb; //500; /*  batch size; <=TRAIN_IMGTOTAL  input samples (numbers + teacher value) for each batch training */

        bool gradient_checked=false;

        /* Timer */
        time_t t_start, t_end;
        int hours,mins,secs;

/*  <<<<<<<<<<<<<<<<<  Read data from MNIST database file  >>>>>>>>>>>>>  */

  	////////////* Read MNIST handwriten digit database *////////////

        /* **   	---- Image data ---
         *
          [offset]   [type]          [value]             [description]
          0000     32bit integer   0x00000803(2051)     magic number
          0004     32bit integer   10000                number of images
          0008     32bit integer   28                   number of rows
          0012     32bit integer   28                   number of columns
          (followings are all pixel data -----: pdata )
          0016     unsigned byte   ??                   pixel
          0017     unsigned byte   ??                   pixel
          ... ...
         */

        /* **   	---- Label data ---
         *
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

	/* ------------ Read in data for training ------------ */

        /* M1. Open MNIST image data file */
	//fd0=open("t10k-images.idx3-ubyte", O_RDONLY);
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
        //fd1=open("t10k-labels.idx1-ubyte", O_RDONLY);
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
//		printf("test_target[%d]: %d\n", i, train_target[i]);
	}

  	////////////* END: Read MNIST handwriten digit database *//////////


/*  <<<<<<<<<<<<<<<<<  Create CNN(Convolution Neural Network) >>>>>>>>>>>>>  */
	printf("Create CNN model...\n");

        /* 1. Create an input CONV3X3 Layer */
	CONV3X3 *conv3x3=new_conv3x3(numFilters, 28,28, data_input); /* numFilters,  w, h, double *din */
	NVLAYER *conv_layer=new_nvlayer(0, NULL, false); /* An empty nvlayer to hold conv3x3 */
	conv_layer->conv3x3=conv3x3;

	/* 2. Create a MAXPOOL2X2 Layer */ /* (CONV3X3 *pinconv3x3, numFilters, imw, imh,  double **din) */
	//MAXPOOL2X2 *maxpool2x2 =new_maxpool2x2( conv3x3, 8, conv3x3->ow, conv3x3->oh, NULL); /* If conv3x3, other's are ignored */
	MAXPOOL2X2 *maxpool2x2 =new_maxpool2x2( conv3x3, 0, 0, 0, NULL); /* If conv3x3, other's are ignored */
	NVLAYER *maxpool_layer=new_nvlayer(0, NULL, false); /* An empty nvlayer to hold conv3x3 */
	maxpool_layer->maxpool2x2=maxpool2x2;

	/* 3. Create the output nvcell layer */
	/*  (nin, NVCELL * const *incells, double *din, double *dw, double bias, double (*transfer)() ) */
	NVCELL *output_tempcell=new_nvcell(maxpool2x2->nf*maxpool2x2->ow*maxpool2x2->oh, NULL, &maxpool2x2->douts[0][0], NULL, 0, NULL); //func_ReLU);
        NVLAYER *output_layer=new_nvlayer(10, output_tempcell, true); /* true for transfunc defined */
        output_layer->transfunc = func_softmax;

        /* 4. Create an nerve net */
        NVNET *nnet=new_nvnet(3); /* 3 layers inside */
        nnet->nvlayers[0]=conv_layer;
        nnet->nvlayers[1]=maxpool_layer;
        nnet->nvlayers[2]=output_layer;

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

        /* 7a. Start timing */
        t_start=time(NULL);
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
                        //printf("\n    === %dth_train, batch item %d/%d ===\n", count+1, i+1, bs);

                        /* 8.2.R1. update ONE sample for input: data_input, data_target */
			#if !BUFFER_IMGDATA /* --- CORSS_CHECK T.1 --- */
                        for(k=0; k<28*28; k++)
				data_input[k]=pTrainImg[(nb*bs+i)*(28*28)+k]/255.0; /* From mmap */
			#else /* --- CORSS_CHECK T.1 --- */
			conv_layer->conv3x3->din = train_imgdata+(nb*bs+i)*(28*28);  /* From pointer train_imgdata */
			#endif
                        for(k=0; k<10; k++)
                                data_target[k] = (k==train_target[nb*bs+i] ? 1.0 : 0.0);

                        /* 8.2.R2. nvnet feed forward, accumlate err values for batch error
                           NOTICE: The output layer has transfunc defined, as func_softmax().
                         */
                        err = nvnet_feed_forward(nnet, data_target, func_lossCrossEntropy); //func_lossMSE);
                        if(isnan(err) || isinf(err) ) { /* If NAN */
                                printf("Return err is nan or inf! Too big learn_rate? or input data unnormalized?\n");
                                exit(1);
                        }
                        batch_err +=err;
                        //printf("nvnet_feed_forward get err=%f\n", err);

                        /* 8.2.R3. nvnet feed backward, update cell->derrs */
                        nvnet_feed_backward(nnet);

                        /* 8.2.R4. update params after feedback(backpropagation) computation */
                        //nvnet_mmtupdate_params(nnet, 0.002); //0.01);
                        nvnet_update_params(nnet, 0.01); /* 0.002 learn_rate */

                    } /* for(i) */

                } /* for(nb) */

                /* 8.3 Mean err for batch training */
                mean_err = batch_err/(nb*bs);

                /* 8.4 Count training */
                count++;

                printf("Epoch %d: samples=%d, mean_err=%0.8f \n",count, TRAIN_IMGTOTAL, mean_err);

        }

        /* 8a. End timing */
        t_end=time(NULL);
        secs=difftime(t_end, t_start);
        hours=secs/3600;
        mins=(secs-hours*3600)/60;
        secs -= 3600*hours+60*mins;

        /* 9. Print results */
        printf("        %dth training, mean_err=%0.8f \n",count, mean_err);
        printf("Finish %d times batch learning! time eplapsed: %02d:%02d:%02d \n",count, hours, mins, secs);


/*  <<<<<<<<<<<<<<<<<  Test CNN Model  >>>>>>>>>>>>>  */

        int errcnt=0;
        printf("\n----------- Test learned NN Model -----------\n");
        for(i=0; i<TEST_IMGTOTAL; i++)
        {
                /* T1. Update data_input,data_target */
		#if !BUFFER_IMGDATA  /* --- CORSS_CHECK 8.2.R1 --- */
                for(k=0; k<28*28; k++)
			 data_input[k]=pTestImg[i*(28*28)+k]/255.0;   /* From mmap */
		#else  /* --- CORSS_CHECK 8.2.R1 --- */
		conv_layer->conv3x3->din = test_imgdata+i*(28*28);    /* Pointer to test_imgdata */
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
                for(k=0; k< output_layer->nc ; k++)
                        printf("%f, ",output_layer->douts[k]);
                printf("    ");
                printf("Target: %d ", test_target[i]);
                if(output_layer->douts[ test_target[i] ] >0.5)
                        printf(" OK\n");
                else {
                        printf(" Fails\n");
                        errcnt++;
                }
        }
	printf("Train samples: %d, train time: %02d:%02d:%02d,  Test samples: %d\n", TRAIN_IMGTOTAL, hours, mins, secs, TEST_IMGTOTAL);
        printf("Err/Total: %d/%d   Accuracy: %.2f%%\n", errcnt, TEST_IMGTOTAL, 100.0*(1.0-1.0*errcnt/TEST_IMGTOTAL));


	/* Free data and nvcell/nvnet */
	free(train_imgdata);
	free(test_imgdata);
	free_nvcell(output_tempcell);
        free_nvnet(nnet); /* free nvnet also free its nvlayers and nvcells inside */

        /* Unmap and close */
        munmap(addr0, 8*1024*1024);
        close(fd0);
        munmap(addr1, 8*1024);
        close(fd1);

	return 0;
}
