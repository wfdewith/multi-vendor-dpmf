#!/bin/sh

set -u

$LIBFM \
	-task r \
	-train $1.train \
	-test $1.test \
	-dim 0,0,5 \
	-iter 256 \
	-method mcmc \
	-init_stdev 0.1 \
	-verbosity 1 \
	-meta $1.group \
	-rlog $1.log
