
#export STAGING_DIR=/home/midas-zhou/openwrt_widora/staging_dir
#COMMON_USRDIR=/home/midas-zhou/openwrt_widora/staging_dir/target-mipsel_24kec+dsp_uClibc-0.9.33.2/usr/

#CC= $(STAGING_DIR)/toolchain-mipsel_24kec+dsp_gcc-4.8-linaro_uClibc-0.9.33.2/bin/mipsel-openwrt-linux-gcc
CC = gcc
CFLAGS += -Wall -O2

#Default set app
TEST_NAME = test_nnc2

###             ----- A template for making test app -----
###     Usage example: make test TEST_NAME=test_conv
###
test:   $(TEST_NAME).c nnc.o actfs.o
	 $(CC) $(CFLAGS) nnc.o actfs.o -lm $(TEST_NAME).c -o $(TEST_NAME)

test_nnc:	test_nnc.c nnc.o actfs.o
	$(CC) $(CFLAGS) nnc.o actfs.o -lm test_nnc.c -o test_nnc

nnc.o:	nnc.c nnc.h
	$(CC) $(CFLAGS) -c nnc.c

actfs.o: actfs.c actfs.h
	$(CC) $(CFLAGS) -c actfs.c

all:

clean:
	rm -rf *.o  test_nnc test_nnc2

