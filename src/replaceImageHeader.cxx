#include "replaceImageHeaderCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkVectorRescaleIntensityImageFilter.h>
#include <itkMinimumMaximumImageFilter.h>
#include <itkImageRegionIterator.h>

#include <iostream>
#include <limits> 

using namespace std;


static const int Dimension = 3;
typedef unsigned short InPixelType;
typedef itk::Image<InPixelType, Dimension> InputImageType;
typedef itk::ImageFileReader<InputImageType> InputImageFileReaderType;
typedef itk::ImageRegionIterator<InputImageType> InputImageRegionIteratorType;

typedef itk::ImageFileWriter<InputImageType> ImageFileWriterType;

int main (int argc, char * argv[]){


	PARSE_ARGS;

	if(inputFilename.compare("") == 0 || outputFilename.compare("") == 0 || referenceFilename.compare("") == 0 ){
		cout<<"Type "<<argv[0]<<" --help to find out how to use this program."<<endl;
		return 1;
	}

	cout<<"Reading: "<<inputFilename<<endl;

	InputImageFileReaderType::Pointer reader = InputImageFileReaderType::New();
	reader->SetFileName(inputFilename);
	reader->Update();
	InputImageType::Pointer inputimage = reader->GetOutput();

	cout<<"Reading: "<<referenceFilename<<endl;

	InputImageFileReaderType::Pointer reader2 = InputImageFileReaderType::New();
	reader2->SetFileName(referenceFilename);
	reader2->Update();
	InputImageType::Pointer refimage = reader2->GetOutput();
	


	InputImageRegionIteratorType it(inputimage, inputimage->GetLargestPossibleRegion());
	it.GoToBegin();

	InputImageRegionIteratorType refit(refimage, refimage->GetLargestPossibleRegion());
	refit.GoToBegin();

	while(!it.IsAtEnd()){
		InputImageType::PixelType pix = it.Get();
		refit.Set(pix);

		++it;
		++refit;
	}
	
	cout<<"Writing: "<<outputFilename<<endl;
	ImageFileWriterType::Pointer writer = ImageFileWriterType::New();
	writer->SetFileName(outputFilename);
	writer->SetInput(refimage);
	writer->Update();



	return 0;
}