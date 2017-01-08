#include "extractMSLesionCLP.h"

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
typedef itk::ImageRegionIterator<InputImageType> InputRegionIteratorType;
typedef itk::NeighborhoodIterator<InputImageType> InputIteratorType;
typedef itk::ImageRandomNonRepeatingConstIteratorWithIndex<InputImageType> InputRandomIteratorType;
typedef itk::ImageFileWriter<InputImageType> ImageFileWriterType;


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
		for(int j = 0; j < VectorPixelDimension; j++){
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

bool containsLabel(InputIteratorType* init, InputImageIndexType index, int labelValueContains, double labelValueContainsPercentageMax){
	if(labelValueContains == -1){
		return true;
	}

	int count = 0;
	int size = init->Size();

	init->SetLocation(index);
	for(int i = 0; i < size; i++){
		if(init->GetPixel(i) == labelValueContains){
			count++;	
		}
	}
	double ratio = ((double)count)/((double)size);
	//cout<<count<<","<<size<<","<<ratio<<endl;
	return  ratio <= labelValueContainsPercentageMax && count > 0;
}

int containsLabel(InputIteratorType* init, int label){

	int size = init->Size();
	int contains = 0;
	for(int i = 0; i < size; i++){
		if(init->GetPixel(i) == label){
			contains++;	
		}
	}
	return contains;
}

int main (int argc, char * argv[]){


	PARSE_ARGS;
	
	if(vectorImageFilename.size() == 0 || (labelImageFilename.compare("") == 0 && index.size() == 0)){
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

	VectorImageType::Pointer vectorimage = composeImageFilter->GetOutput();

	if(outFileNameComposed.compare("") != 0){
		cout<<"Writing file: "<<outFileNameComposed<<endl;
		VectorImageFileWriterType::Pointer writer = VectorImageFileWriterType::New();
		writer->SetFileName(outFileNameComposed);
		writer->SetInput(vectorimage);
		writer->Update();
	}

	InputImageType::Pointer labelimage = 0;

	if(labelImageFilename.compare("") != 0){
		InputImageFileReaderType::Pointer readimage = InputImageFileReaderType::New();
		readimage->SetFileName(labelImageFilename);
		readimage->Update();
		labelimage = readimage->GetOutput();	
	}	

	InputImageType::Pointer frequencymap = 0;

	if(frequencyMapFilename.compare("") != 0){
		cout<<"Reading frequencymap: "<<frequencyMapFilename<<endl;
		InputImageFileReaderType::Pointer readimage = InputImageFileReaderType::New();
		readimage->SetFileName(frequencyMapFilename);
		readimage->Update();
		frequencymap = readimage->GetOutput();
	}

	VectorImageIteratorType::RadiusType radius;	
	radius[0] = neighborhood[0];
	radius[1] = neighborhood[1];
	radius[2] = neighborhood[2];

	VectorImageType::Pointer outputimage = VectorImageType::New();

	VectorImageType::SizeType size;
	size[0] = radius[0]*2 + 1;
	size[1] = radius[1]*2 + 1;
	size[2] = radius[2]*2 + 1;
	VectorImageType::RegionType region;
	region.SetSize(size);
	
	outputimage->SetRegions(region);
	outputimage->SetVectorLength(vectorImageFilename.size());
	outputimage->SetSpacing(vectorimage->GetSpacing());
	outputimage->SetDirection(vectorimage->GetDirection());
	outputimage->Allocate();

	InputImageType::Pointer outputimagelabel = InputImageType::New();
	outputimagelabel->SetRegions(region);
	outputimagelabel->SetSpacing(vectorimage->GetSpacing());
	outputimagelabel->SetDirection(vectorimage->GetDirection());
	outputimagelabel->Allocate();

	VectorImageIteratorType vectorit(radius, vectorimage, vectorimage->GetLargestPossibleRegion());

	if(refImageFilename.compare("") != 0){

		typedef itk::ImageFileReader<VectorImageType> VectorImageFileReaderType;
		VectorImageFileReaderType::Pointer refreader = VectorImageFileReaderType::New();
		refreader->SetFileName(refImageFilename);
		refreader->Update();

		VectorImageType::Pointer refimage = refreader->GetOutput();

		VectorImageType::PointType reforigin = refimage->GetOrigin();
		VectorImageType::IndexType vecindex;

		vectorimage->TransformPhysicalPointToIndex(reforigin, vecindex);

		vecindex[0] += neighborhood[0];
		vecindex[1] += neighborhood[1];
		vecindex[2] += neighborhood[2];

		vectorit.SetLocation(vecindex);

		VectorImageIteratorType outit(radius, outputimage, outputimage->GetLargestPossibleRegion());
		outit.GoToBegin();

		outputimage->SetOrigin(reforigin);

		InputIteratorType init(radius, labelimage, labelimage->GetLargestPossibleRegion());
		init.SetLocation(vecindex);

		if(!containsLabel(&init, labelValue)){
			int i = 0;
			while(!outit.IsAtEnd()){
				outit.SetCenterPixel(vectorit.GetPixel(i));				
				++outit;
				i++;
			}

			cout<<"Writing file: "<<outputFileName<<endl;
			VectorImageFileWriterType::Pointer writer = VectorImageFileWriterType::New();
			writer->SetFileName(outputFileName);
			writer->SetInput(outputimage);
			writer->Update();
		}else{
			cerr<<"The region contains label: "<<labelValue<<endl;
		}

	}else if(index.size() == 0){
		
		InputRandomIteratorType randomit(labelimage, labelimage->GetLargestPossibleRegion());
		randomit.SetNumberOfSamples(labelimage->GetLargestPossibleRegion().GetNumberOfPixels());
		randomit.GoToBegin();

		InputIteratorType init(radius, labelimage, labelimage->GetLargestPossibleRegion());

		char *uuid = new char[100];	

		cout<<"Searching for label: "<<labelValue<<endl;
		if(labelValueContains != -1){
			cout<<"The region contains label: "<<labelValueContains<<", ratiomax: "<<labelValueContainsPercentageMax<<endl;
		}	

		InputRegionIteratorType itfreq;
		if(frequencymap){

			itfreq = InputRegionIteratorType(frequencymap, frequencymap->GetLargestPossibleRegion());
		}
		
		
		while(!randomit.IsAtEnd() && numberOfSamples){
			if(randomit.Get() == labelValue && containsLabel(&init, randomit.GetIndex(), labelValueContains, labelValueContainsPercentageMax)){
				bool testfreq = true;
				if(frequencymap){
					itfreq.SetIndex(randomit.GetIndex());
					testfreq = itfreq.Get() >= frequencyMapTreshold;
				}
				if(testfreq){
					
					uuid_t id;
					uuid_generate(id);
				  	uuid_unparse(id, uuid);

				  	numberOfSamples--;

				  	string outfilename = outputImageDirectory;
				  	outfilename.append(string(uuid)).append(".nrrd");

				  	// if(vectorimage->GetVectorLength() == 2){
				  	// 	writeImage<2>(&vectorit, radius, outfilename);
				  	// }else if(vectorimage->GetVectorLength() == 3){
				  	// 	writeImage<3>(&vectorit, radius, outfilename);
				  	// }else{
				  	// 	throw "Modify the source code here and recompile to use with the appropriate number of components.";
				  	// }

					VectorImageIteratorType outit(radius, outputimage, outputimage->GetLargestPossibleRegion());
					outit.GoToBegin();

					InputIteratorType outitlabel(radius, outputimagelabel, outputimagelabel->GetLargestPossibleRegion());
					outitlabel.GoToBegin();

					vectorit.SetLocation(randomit.GetIndex());
					init.SetLocation(randomit.GetIndex());


					InputImageType::PointType outorigin;

					VectorImageType::IndexType outputindex = randomit.GetIndex();
					outputindex[0] -= neighborhood[0];
					outputindex[1] -= neighborhood[1];
					outputindex[2] -= neighborhood[2];

					vectorimage->TransformIndexToPhysicalPoint(outputindex, outorigin);

					outputimage->SetOrigin(outorigin);

					int i = 0;
					while(!outit.IsAtEnd()){
						outit.SetCenterPixel(vectorit.GetPixel(i));

						outitlabel.SetCenterPixel(init.GetPixel(i));
						init.SetPixel(i, 0);//Mark the adjacent voxels as visited and avoid selecting same lesion again

						++outit;
						i++;
					}

					cout<<"Writing file: "<<outfilename<<endl;
					VectorImageFileWriterType::Pointer writer = VectorImageFileWriterType::New();
					writer->SetFileName(outfilename);
					writer->SetInput(outputimage);
					writer->Update();

					if(writeLabel){
						string outfilenamelabel = outputImageDirectory;
				  		outfilenamelabel.append(string(uuid)).append("_label.nrrd");
					}
				}
			}
			++init;
			++vectorit;
			++randomit;
		}

		delete[] uuid;

	}else{
		VectorImageType::IndexType vectorindex;
		vectorindex[0] = index[0];
		vectorindex[1] = index[1];
		vectorindex[2] = index[2];

		vectorit.SetLocation(vectorindex);

		VectorImageIteratorType outit(radius, outputimage, outputimage->GetLargestPossibleRegion());
		outit.GoToBegin();

		InputIteratorType init; 

		if(labelimage){
			init = InputIteratorType(radius, labelimage, labelimage->GetLargestPossibleRegion());
			if(!init.GetCenterPixel() == labelValue && containsLabel(&init, init.GetIndex(), labelValueContains, labelValueContainsPercentageMax)){
				int i = 0;
				while(!outit.IsAtEnd()){
					outit.SetCenterPixel(vectorit.GetPixel(i));				
					++outit;
					i++;
				}

				cout<<"Writing file: "<<outputFileName<<endl;
				VectorImageFileWriterType::Pointer writer = VectorImageFileWriterType::New();
				writer->SetFileName(outputFileName);
				writer->SetInput(outputimage);
				writer->Update();
			}else{
				cerr<<"Label not found."<<endl;
			}
		}else{
			int i = 0;
			while(!outit.IsAtEnd()){
				outit.SetCenterPixel(vectorit.GetPixel(i));				
				++outit;
				i++;
			}

			cout<<"Writing file: "<<outputFileName<<endl;
			VectorImageFileWriterType::Pointer writer = VectorImageFileWriterType::New();
			writer->SetFileName(outputFileName);
			writer->SetInput(outputimage);
			writer->Update();
		}
		
	}
	

	


	return 0;
}