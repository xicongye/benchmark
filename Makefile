all: coremark-build dhrystone-build stream-build
	rm -rf benchmark_bin
	mkdir benchmark_bin
	mv coremark/coremark-O* benchmark_bin
	mv dhrystone/dhrystone-O* benchmark_bin
	mv STREAM/stream_c.exe benchmark_bin

coremark-build:
	make -C coremark clean
	make -C coremark

dhrystone-build:
	make -C dhrystone clean
	make -C dhrystone

stream-build:
	make -C STREAM clean
	make -C STREAM

clean:
	rm -rf benchmark_bin
	make -C coremark clean
	make -C dhrystone clean
	make -C STREAM clean

