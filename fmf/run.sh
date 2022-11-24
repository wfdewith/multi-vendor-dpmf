#!/usr/bin/zsh

DIR=$(pwd)

pushd data
for n in 10 40; do
	let "m = n - 1"
	pushd r$n
	for i in {0..$m}; do
		sem -j-2 "$DIR/do.sh $i"
	done
	popd
done
popd
sem --wait
