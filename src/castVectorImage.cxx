#include "castVectorImageCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkVectorRescaleIntensityImageFilter.h>

#include <itkImageRegionIterator.h>

#include <iostream>
#include <limits> 

using namespace std;


static const int Dimension = 3;
static const int InputPixelDimension = 2;
typedef unsigned short InPixelType;
typedef itk::CovariantVector<InPixelType, InputPixelDimension> InputPixelType;
typedef itk::Image<InputPixelType, Dimension> InputImageType;
typedef itk::ImageFileReader<InputImageType> InputImageFileReaderType;


static const int OutputPixelDimension = 4;
typedef unsigned char OutPixelType;
typedef itk::CovariantVector<OutPixelType, OutputPixelDimension> OutputPixelType;
typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
typedef itk::ImageFileWriter<OutputImageType> OutputImageFileWriterType;
typedef itk::ImageRegionIterator<OutputImageType> OutputImageRegionIteratorType;




int main (int argc, char * argv[]){


	PARSE_ARGS;

	if(inputFilename.compare("") == 0){
		cout<<"Type "<<argv[0]<<" --help to find out how to use this program."<<endl;
		return 1;
	}

	InputImageFileReaderType::Pointer reader = InputImageFileReaderType::New();
	reader->SetFileName(inputFilename);
	reader->Update();
	InputImageType::Pointer inputimage = reader->GetOutput();

	typedef itk::VectorRescaleIntensityImageFilter<InputImageType, OutputImageType> VectorRescaleIntensityImageFilterType;
	VectorRescaleIntensityImageFilterType::Pointer vectorrescale = VectorRescaleIntensityImageFilterType::New();
	vectorrescale->SetOutputMaximumMagnitude(numeric_limits<OutPixelType>::max());
	vectorrescale->SetInput(inputimage);
	vectorrescale->Update();

	
	OutputImageType::Pointer outputimage = vectorrescale->GetOutput();

	OutputImageRegionIteratorType outit(outputimage, outputimage->GetLargestPossibleRegion());

	while(!outit.IsAtEnd()){
		OutputPixelType pix = outit.Get();
		pix[2] = 0;
		pix[3] = 255;
		outit.Set(pix);
		++outit;
	}
	
	OutputImageFileWriterType::Pointer writer = OutputImageFileWriterType::New();
	writer->SetFileName(outputFilename);
	writer->SetInput(outputimage);
	writer->Update();



	return 0;
}