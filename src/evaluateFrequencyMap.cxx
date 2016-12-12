#include "evaluateFrequencyMapCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkComposeImageFilter.h>
#include <itkVectorImage.h>
#include <itkImageFileWriter.h>
#include <itkNeighborhoodIterator.h>
#include <itkImageRegionIterator.h>
#include <itkImageRandomNonRepeatingIteratorWithIndex.h>
#include <itkFixedArray.h>

#include <uuid/uuid.h>
#include <iostream>

#include <itkConnectedComponentImageFilter.h>
#include <itkThresholdImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkLabelImageToLabelMapFilter.h>
#include <itkLabelStatisticsImageFilter.h>

using namespace std;

typedef unsigned short PixelType;
static const int Dimension = 3;

typedef itk::Image<PixelType, Dimension> InputImageType;
typedef InputImageType::IndexType InputImageIndexType;
typedef itk::ImageFileReader<InputImageType> InputImageFileReaderType;

typedef unsigned short VectorImagePixelType;
typedef itk::VectorImage<VectorImagePixelType, Dimension> VectorImageType;  
typedef itk::ImageFileReader<VectorImageType> VectorImageFileReaderType;
typedef VectorImageType::PointType VectorImagePointType;


int main (int argc, char * argv[]){


	PARSE_ARGS;
	
	if(refImageFilename.size() == 0 || frequencyMapFilename.compare("") == 0){
		cerr<<"Type "<<argv[0]<<" --help to find out how to use this program."<<endl;
		return EXIT_FAILURE;
	}
	
	InputImageFileReaderType::Pointer readimage = InputImageFileReaderType::New();
	readimage->SetFileName(frequencyMapFilename);
	readimage->Update();
	InputImageType::Pointer frequencymap = readimage->GetOutput();
	

	typedef itk::ImageFileReader<VectorImageType> VectorImageFileReaderType;
	VectorImageFileReaderType::Pointer vectreader = VectorImageFileReaderType::New();
	vectreader->SetFileName(refImageFilename);
	vectreader->Update();

	VectorImageType::Pointer sampleimage = vectreader->GetOutput();
	VectorImagePointType origin = sampleimage->GetOrigin();

	InputImageIndexType index;

	frequencymap->TransformPhysicalPointToIndex(origin, index);

	index[0] += neighborhood[0];
	index[1] += neighborhood[1];
	index[2] += neighborhood[2];

	cout<<frequencymap->GetPixel(index)<<endl;

	return EXIT_SUCCESS;
}