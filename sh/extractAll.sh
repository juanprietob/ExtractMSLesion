#!/bin/bash 
dir=$1
outdir=$2

if [ -z "$dir" ]
then
    dir="./"
fi

if [ -z "$outdir" ]
then
    outdir="./"
fi

echo "Using input dir: "$dir
echo "Using outputdir: "$outdir

scriptdir=${BASH_SOURCE[0]}
scriptname=$(basename ${scriptdir})
scriptdir=${scriptdir%${scriptname}*}
extractMSLesion=${scriptdir}extractMSLesion

echo "Using executable "${extractMSLesion}

function executeMSLesion {
	pvec=$1
	localdir=${pvec%/*}
	pd=${localdir}/PD.nii.gz 
    t2=${localdir}/T2.nii.gz

    extradir=${localdir#${dir}*}

    echo mkdir -p $outdir/ms/${extradir}
    echo $extractMSLesion --labelImg $pvec --img $pd --img $t2 --labelValue 6 --outDir $outdir/ms/${extradir}
    echo mkdir -p $outdir/wm/${extradir}
    echo $extractMSLesion --labelImg $pvec --img $pd --img $t2 --labelValue 8 --labelValueContains 6 --outDir $outdir/wm/${extradir}

} 

find ${dir} -name "pvec.nii.gz" | while read file; do executeMSLesion "$file"; done


