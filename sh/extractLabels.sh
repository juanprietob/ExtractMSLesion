file=$1
type=$2
labelnumber=$3

if [ -z "$file" ]
then
    echo "You must provide a file with all the patient ids in the form 'patientid/date'"
    echo $0 "filename" "type" "labelnumber"
    exit
fi

if [ -z "$type" ]
then
    echo "You must provide the type of output 'ms' 'wm' 'gm'"
    echo $0 "filename" "type" "labelnumber"
    exit
fi

if [ -z "$labelnumber" ]
then
    echo "Label number"
    echo $0 "filename" "type" "labelnumber"
    exit
fi

for pids in $(cat $file); do 
dirs=/Volumes/BookStudio1/data/CLIMB/$pids; 

mkdir -p /Volumes/BookStudio1/data/deepLearningExemplars1/$type/$pids ; 
./extractMSLesion --img $dirs/PD.nii.gz --img $dirs/T2.nii.gz --labelImg $dirs/pvec.nii.gz --labelValue $labelnumber --outDir /Volumes/BookStudio1/data/deepLearningExemplars1/$type/$pids --numberOfSamples 100; 
echo $dirs >> $typedone.txt ;

done;