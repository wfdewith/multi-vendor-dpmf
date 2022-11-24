#!/bin/sh

$LIBFM \
	-task r \
	-train $1.train \
	-test $1.test \
	-dim 1,1,16 \
	-iter 100 \
	-method mcmc \
	-init_stdev 0.1 \
	-verbosity 1 \
	-meta $1.group \
	-rlog $1.log
