
export STAGING_DIR=/home/midas-zhou/openwrt_widora/staging_dir
COMMON_USRDIR=/home/midas-zhou/openwrt_widora/staging_dir/target-mipsel_24kec+dsp_uClibc-0.9.33.2/usr/

#CC= $(STAGING_DIR)/toolchain-mipsel_24kec+dsp_gcc-4.8-linaro_uClibc-0.9.33.2/bin/mipsel-openwrt-linux-gcc
CC = gcc

test_nnc:	test_nnc.c nnc.o
	$(CC)  nnc.o -lm test_nnc.c -o test_nnc

nnc.o:	nnc.c nnc.h
	$(CC) -c nnc.c

clean:
	rm -rf *.o test_nnc

