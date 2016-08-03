dir=$1
for files in $(ls temp/*.png)
do
	outname=$(basename ${files%%.png}Out.eps)
	convert $files $outname
done