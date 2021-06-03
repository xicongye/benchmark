CC = riscv64-unknown-linux-gnu-gcc
SRC = core_list_join.c core_main.c core_matrix.c core_state.c core_util.c core_portme.c
HDR = core_portme.h coremark.h

all: coremark-O2-MULTI-1 \
     coremark-O2-vector-MULTI-1 \
     coremark-O2-MULTI-4 \
     coremark-O2-vector-MULTI-4

# one-hart
MULTI-1-CFLAGS := -DMULTITHREAD=1 -DUSE_PTHREAD -DPERFORMANCE_RUN=1 -lrt -lpthread -DITERATIONS=2000
MULTI-1-LDFLAGS := 

coremark-O2-MULTI-1: $(SRC) $(HDR)
	$(CC) -O2 $(MULTI-1-CFLAGS) -DFLAGS_STR=\""-O2 $(MULTI-1-CFLAGS)"\" $(SRC) $(MULTI-1-LDFLAGS) -o $@

coremark-O2-vector-MULTI-1: $(SRC) $(HDR)
	$(CC) -O2 -DRVV $(MULTI-1-CFLAGS) -DFLAGS_STR=\""-O2 -DRVV $(MULTI-1-CFLAGS)"\" $(SRC) $(MULTI-1-LDFLAGS) -o $@


# four-hart
MULTI-4-CFLAGS := -DMULTITHREAD=4 -DUSE_PTHREAD -DPERFORMANCE_RUN=1 -lrt -lpthread -DITERATIONS=8000
MULTI-4-LDFLAGS := 

coremark-O2-MULTI-4: $(SRC) $(HDR)
	$(CC) -O2 $(MULTI-4-CFLAGS) -DFLAGS_STR=\""-O2 $(MULTI-4-CFLAGS)"\" $(SRC) $(MULTI-4-LDFLAGS) -o $@

coremark-O2-vector-MULTI-4: $(SRC) $(HDR)
	$(CC) -O2 -DRVV $(MULTI-4-CFLAGS) -DFLAGS_STR=\""-O2 -DRVV $(MULTI-4-CFLAGS)"\" $(SRC) $(MULTI-4-LDFLAGS) -o $@

clean:
	rm -f 	coremark-O2-MULTI-1 \
		coremark-O2-vector-MULTI-1 \
		coremark-O2-MULTI-4 \
		coremark-O2-vector-MULTI-4

