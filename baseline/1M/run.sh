#!/usr/bin/zsh

DIR=$(pwd)

pushd data
for n in {1..10}; do
	let "m = n - 1"
	pushd r$n
	for i in {0..$m}; do
		pushd p$i
		for j in {0..9}; do
			sem -j+0 "$DIR/do.sh $j $i $m"
		done
		popd
	done
	popd
done
popd
sem --wait
