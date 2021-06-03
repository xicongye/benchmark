DHRY-CFLAGS := -DTIMES -Wno-implicit -w
DHRY-CFLAGS += -DHZ=100

CC = riscv64-unknown-linux-gnu-gcc
DHRY-CFLAGS += -DDHRY_ITERS=2000000

SRC = dhry_1.c dhry_2.c
HDR = dhry.h

CFLAGS := $(DHRY-CFLAGS) $(XCFLAGS)

all: dhrystone-O2 dhrystone-O3 dhrystone-O3-flto
dhrystone-O2: $(SRC) $(HDR)
	$(CC) -O2 $(CFLAGS) $(SRC) $(LDFLAGS) $(LOADLIBES) $(LDLIBS) -o $@

dhrystone-O3: $(SRC) $(HDR)
	$(CC) -O3 $(CFLAGS) $(SRC) $(LDFLAGS) $(LOADLIBES) $(LDLIBS) -o $@

dhrystone-O3-flto: $(SRC) $(HDR)
	$(CC) -O3 -flto $(CFLAGS) $(SRC) $(LDFLAGS) $(LOADLIBES) $(LDLIBS) -o $@

clean:
	rm -f dhrystone-O2 dhrystone-O3 dhrystone-O3-flto


