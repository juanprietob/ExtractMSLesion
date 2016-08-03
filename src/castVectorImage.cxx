#include "castVectorImageCLP.h"

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
static const int InputPixelDimension = 2;
typedef unsigned short InPixelType;
typedef itk::CovariantVector<InPixelType, InputPixelDimension> InputPixelType;
typedef itk::Image<InputPixelType, Dimension> InputImageType;
typedef itk::ImageFileReader<InputImageType> InputImageFileReaderType;
typedef itk::ImageRegionIterator<InputImageType> InputImageRegionIteratorType;

static const int OutputPixelDimension = 4;
typedef unsigned char OutPixelType;
typedef itk::FixedArray<OutPixelType, OutputPixelDimension> OutputPixelType;
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
	InputImageType::PixelType minpix;
	minpix.Fill(numeric_limits<InPixelType>::max());

	InputImageType::PixelType maxpix;
	maxpix.Fill(numeric_limits<InPixelType>::min());


	InputImageRegionIteratorType it(inputimage, inputimage->GetLargestPossibleRegion());

	it.GoToBegin();

	while(!it.IsAtEnd()){
		InputImageType::PixelType pix = it.Get();
		for(int i = 0; i < pix.Size(); i++){
			if(pix[i] < minpix[i]){
				minpix[i] = pix[i];
			}
			if(pix[i] > maxpix[i]){
				maxpix[i] = pix[i];
			}
		}
		++it;
	}

	InPixelType maxmax = numeric_limits<InPixelType>::min();
	for(int i = 0; i < maxpix.Size(); i++){
		maxmax = max(maxmax, maxpix[i]);
	}

	InPixelType minmin = numeric_limits<InPixelType>::max();
	for(int i = 0; i < minpix.Size(); i++){
		minmin = min(minmin, minpix[i]);
	}


	
	//OutputImageType::Pointer outputimage = vectorrescale->GetOutput();

	OutputImageType::Pointer outputimage = OutputImageType::New();
	outputimage->SetRegions(inputimage->GetLargestPossibleRegion());
	outputimage->Allocate();
	OutputImageRegionIteratorType outit(outputimage, outputimage->GetLargestPossibleRegion());

	outit.GoToBegin();
	it.GoToBegin();
	while(!outit.IsAtEnd()){
		InputPixelType inpix = it.Get();
		OutputPixelType outpix = outit.Get();
		outpix[0] = 255.0*(inpix[0] - minpix[0])/(maxpix[0] - minpix[0]);
		outpix[1] = 255.0*(inpix[1] - minpix[1])/(maxpix[1] - minpix[1]);
		outpix[2] = 0;
		outpix[3] = 255;

		
		outit.Set(outpix);
		++outit;
		++it;
	}
	
	OutputImageFileWriterType::Pointer writer = OutputImageFileWriterType::New();
	writer->SetFileName(outputFilename);
	writer->SetInput(outputimage);
	writer->Update();



	return 0;
}