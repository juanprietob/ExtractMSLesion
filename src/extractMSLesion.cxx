#include "extractMSLesionCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkComposeImageFilter.h>
#include <itkVectorImage.h>
#include <itkImageFileWriter.h>
#include <itkNeighborhoodIterator.h>
#include <itkFixedArray.h>

#include <uuid/uuid.h>
#include <iostream>

using namespace std;

typedef unsigned short PixelType;
static const int Dimension = 3;

typedef itk::Image<PixelType, Dimension> InputImageType;
typedef itk::ImageFileReader<InputImageType> InputImageFileReaderType;
typedef itk::NeighborhoodIterator<InputImageType> InputIteratorType;

typedef unsigned short VectorImagePixelType;
typedef itk::VectorImage<VectorImagePixelType, Dimension> VectorImageType;  
typedef itk::ComposeImageFilter< InputImageType, VectorImageType> ComposeImageFilterType;
typedef itk::NeighborhoodIterator<VectorImageType> VectorImageIteratorType;
typedef VectorImageIteratorType::RadiusType VectorImageRadiusType;
typedef itk::ImageFileWriter<VectorImageType> VectorImageFileWriterType;

template <unsigned int VectorPixelDimension>
void writeImage(VectorImageIteratorType* vectorit, VectorImageRadiusType radius, string outfilename){

	typedef itk::FixedArray<VectorImagePixelType, VectorPixelDimension> OutputPixelType;
	typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
	typedef itk::ImageRegionIterator<OutputImageType> OutputImageRegionIteratorType;
	typedef itk::ImageFileWriter<OutputImageType> OutputImageFileWriterType;


	typename OutputImageType::Pointer outputimage = OutputImageType::New();
	typename OutputImageType::RegionType region;
	typename OutputImageType::SizeType size;
	size[0] = radius[0]*2 + 1;
	size[1] = radius[1]*2 + 1;
	size[2] = radius[2]*2 + 1;

	region.SetSize(size);

	OutputPixelType defpixel;
	defpixel.Fill(0);
	
	outputimage->SetRegions(region);
	outputimage->Allocate();
	outputimage->FillBuffer(defpixel);	
	

	OutputImageRegionIteratorType outit(outputimage, outputimage->GetLargestPossibleRegion());
	outit.GoToBegin();

	int i = 0;
	while(!outit.IsAtEnd()){
		VectorImageType::PixelType pix = vectorit->GetPixel(i);		
		OutputPixelType outpix;
		outpix.Fill(0);
		for(int j = 0; j < 2; j++){
			outpix[j] = pix[j];			
		}
		outit.Set(outpix);
		++outit;
		i++;
	}
  	

	cout<<"Writing file: "<<outfilename<<endl;
	typename OutputImageFileWriterType::Pointer writer = OutputImageFileWriterType::New();
  	writer->SetFileName(outfilename);
  	writer->SetInput(outputimage);
  	writer->Update();
}

bool containsLabel(InputIteratorType* init, int labelValueContains, double labelValueContainsPercentage){
	if(labelValueContains == -1){
		return true;
	}

	int count = 0;
	int size = init->Size();

	for(int i = 0; i < size; i++){
		if(init->GetPixel(i) == labelValueContains){
			count++;	
		}
	}
	double ratio = ((double)count)/((double)size);
	return  ratio >= labelValueContainsPercentage;
}

int main (int argc, char * argv[]){


	PARSE_ARGS;
	
	if(vectorImageFilename.size() == 0 || labelImageFilename.compare("") == 0){
		cout<<"Type "<<argv[0]<<" --help to find out how to use this program."<<endl;
		return 1;
	}

	if(outputImageDirectory.compare("") != 0){
		outputImageDirectory.append("/");
	}

	ComposeImageFilterType::Pointer composeImageFilter = ComposeImageFilterType::New();

	for(int i = 0; i < vectorImageFilename.size(); i++){
		cout<<"Reading:"<<vectorImageFilename[i]<<endl;
		InputImageFileReaderType::Pointer readimage = InputImageFileReaderType::New();
		readimage->SetFileName(vectorImageFilename[i]);
		readimage->Update();

		composeImageFilter->SetInput(i, readimage->GetOutput());
	}

	composeImageFilter->Update();

	InputImageFileReaderType::Pointer readimage = InputImageFileReaderType::New();
	readimage->SetFileName(labelImageFilename);
	readimage->Update();

	InputImageType::Pointer labelimage = readimage->GetOutput();	

	VectorImageType::Pointer vectorimage = composeImageFilter->GetOutput();

	VectorImageIteratorType::RadiusType radius;	
	radius[0] = neighborhood[0];
	radius[1] = neighborhood[1];
	radius[2] = neighborhood[2];

	InputIteratorType init(radius, labelimage, labelimage->GetLargestPossibleRegion());
	init.GoToBegin();
	

	char *uuid = new char[100];	


	VectorImageIteratorType vectorit(radius, vectorimage, vectorimage->GetLargestPossibleRegion());

	cout<<"Searching for label: "<<labelValue<<endl;
	if(labelValueContains != -1){
		cout<<"The region contains label: "<<labelValueContains<<", ratio: "<<labelValueContainsPercentage<<endl;
	}
	
	vectorit.GoToBegin();
	while(!init.IsAtEnd() && !vectorit.IsAtEnd()){
		if(init.GetCenterPixel() == labelValue && containsLabel(&init, labelValueContains, labelValueContainsPercentage)){
			uuid_t id;
			uuid_generate(id);
		  	uuid_unparse(id, uuid);

		  	string outfilename = outputImageDirectory;
		  	outfilename.append(string(uuid)).append(".nrrd");

		  	writeImage<2>(&vectorit, radius, outfilename);

			// VectorImageType::Pointer outputimage = VectorImageType::New();
			// VectorImageType::RegionType region;
			// VectorImageType::SizeType size;
			
			// size[0] = radius[0]*2 + 1;
			// size[1] = radius[1]*2 + 1;
			// size[2] = radius[2]*2 + 1;

			// region.SetSize(size);			
			
			// outputimage->SetRegions(region);
			// outputimage->SetVectorLength(2);
			// outputimage->Allocate();
			

			// VectorImageIteratorType outit(radius, outputimage, outputimage->GetLargestPossibleRegion());
			// outit.GoToBegin();

			// int i = 0;
			// while(!outit.IsAtEnd()){
			// 	outit.SetCenterPixel(vectorit.GetPixel(i));				
			// 	++outit;
			// 	i++;
			// }

			// VectorImageFileWriterType::Pointer writer = VectorImageFileWriterType::New();
			// writer->SetFileName(outfilename);
			// writer->SetInput(outputimage);
			// writer->Update();
		}
		++init;
		++vectorit;
	}

	delete[] uuid;


	return 0;
}