#!/usr/bin/zsh

DIR=$(pwd)

pushd data_full
sem -j-2 "$DIR/do.sh 0"
popd

pushd data_vendors
for n in {1..10}; do
	pushd r$n
	for i in {0..9}; do
		sem -j-2 "$DIR/do.sh $i"
	done
	popd
done
popd
sem --wait
