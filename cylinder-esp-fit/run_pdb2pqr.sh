ver=amber
pdb2pqr \
	--ff=$ver \
	--ffout=$ver \
	--chain \
	--apbs-input \
	--verbose \
	--with-ph=7.4 \
	--summary \
	./t4-tail-five-disc-extra-tube.pdb ./t4-tail-five-disc-extra-tube.$ver.pqr > out.$ver.five-extra-tube.log
