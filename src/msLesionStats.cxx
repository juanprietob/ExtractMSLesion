#include "msLesionStatsCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkLabelImageToLabelMapFilter.h>
#include <itkConnectedThresholdImageFilter.h>
#include <itkLabelStatisticsImageFilter.h>
#include <itkLabelMapMaskImageFilter.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkImageFileWriter.h>

#include <itkComposeImageFilter.h>
#include <itkVectorImage.h>
#include <itkNeighborhoodIterator.h>
#include <uuid/uuid.h>
#include <iostream>

using namespace std;

typedef unsigned short PixelType;
static const int Dimension = 3;

typedef itk::Image<PixelType, Dimension> InputImageType;
typedef InputImageType::IndexType InputImageIndexType;
typedef itk::ImageFileReader<InputImageType> InputImageFileReaderType;


typedef itk::ImageFileWriter<InputImageType> ImageFileWriterType;

typedef unsigned short VectorImagePixelType;
typedef itk::VectorImage<VectorImagePixelType, Dimension> VectorImageType;  
typedef itk::ComposeImageFilter< InputImageType, VectorImageType> ComposeImageFilterType;
typedef itk::NeighborhoodIterator<VectorImageType> VectorImageIteratorType;
typedef VectorImageIteratorType::RadiusType VectorImageRadiusType;
typedef itk::ImageFileWriter<VectorImageType> VectorImageFileWriterType;


int main (int argc, char * argv[]){


	PARSE_ARGS;
	
	if(labelImageFilename.compare("") == 0){
		cout<<"Type "<<argv[0]<<" --help to find out how to use this program."<<endl;
		return 1;
	}

	if(outputImageDirectory.compare("") != 0){
		outputImageDirectory.append("/");
	}

	InputImageFileReaderType::Pointer readimage = InputImageFileReaderType::New();
	readimage->SetFileName(labelImageFilename);
	readimage->Update();

	InputImageType::Pointer labelimage = readimage->GetOutput();

    typedef itk::LabelObject< PixelType, Dimension > LabelObjectType;
    typedef itk::LabelMap< LabelObjectType > LabelImageType;

    typedef itk::LabelMapMaskImageFilter< LabelImageType, InputImageType > LabelMapMaskImageFilterType;

    typedef itk::LabelImageToLabelMapFilter< InputImageType, LabelImageType > LabelImageMapFilterType;
    LabelImageMapFilterType::Pointer labelmapfilter = LabelImageMapFilterType::New();

    labelmapfilter->SetInput(labelimage);
    labelmapfilter->Update();
    LabelImageType::Pointer labelmap = labelmapfilter->GetOutput();

    // LabelImageType::LabelVectorType labels;
    // labels = labelmap->GetLabels();

    // cout<<labels.size()<<endl;

    LabelMapMaskImageFilterType::Pointer labelmaskfilter = LabelMapMaskImageFilterType::New();
    labelmaskfilter->SetInput( labelmap );
    labelmaskfilter->SetFeatureImage( labelimage );
    labelmaskfilter->SetBackgroundValue( 0 );
    labelmaskfilter->SetLabel(labelValue);
    labelmaskfilter->Update();

    typedef itk::ConnectedComponentImageFilter <InputImageType, InputImageType > ConnectedComponentImageFilterType;
    typedef ConnectedComponentImageFilterType::Pointer                           ConnectedComponentImageFilterPointerType;

    ConnectedComponentImageFilterPointerType connectedcomponents = ConnectedComponentImageFilterType::New();
	connectedcomponents->SetInput(labelmaskfilter->GetOutput());

	InputImageType::Pointer connectedregionsimage = connectedcomponents->GetOutput();
    

    typedef itk::LabelStatisticsImageFilter< InputImageType, InputImageType >  LabelStatisticsImageFilterType;
	typedef LabelStatisticsImageFilterType::Pointer                    LabelStatisticsImageFilterPointerType;

    LabelStatisticsImageFilterPointerType connectedregionsstatistics = LabelStatisticsImageFilterType::New();

    connectedregionsstatistics->SetLabelInput(connectedregionsimage);
    connectedregionsstatistics->SetInput(connectedregionsimage);
    connectedregionsstatistics->Update();

    typedef LabelStatisticsImageFilterType::ValidLabelValuesContainerType ValidLabelValuesType;
	typedef ValidLabelValuesType::const_iterator ValidLabelsIteratorType;
	typedef LabelStatisticsImageFilterType::LabelPixelType                LabelPixelType;

	ValidLabelValuesType validlabels = connectedregionsstatistics->GetValidLabelValues();


	VectorImageType::Pointer vectorimage = 0;
	if(extractLesion){
		ComposeImageFilterType::Pointer composeImageFilter = ComposeImageFilterType::New();

		for(int i = 0; i < vectorImageFilename.size(); i++){
			cout<<"Reading:"<<vectorImageFilename[i]<<endl;
			InputImageFileReaderType::Pointer readimage = InputImageFileReaderType::New();
			readimage->SetFileName(vectorImageFilename[i]);
			readimage->Update();

			composeImageFilter->SetInput(i, readimage->GetOutput());
		}

		composeImageFilter->Update();
		vectorimage = composeImageFilter->GetOutput();

		
	}

	VectorImageIteratorType::RadiusType radius;	
	radius[0] = neighborhood[0];
	radius[1] = neighborhood[1];
	radius[2] = neighborhood[2];
	

	char *uuid = new char[100];	

	//Get each label and insert it into the list sample
	for(ValidLabelsIteratorType vIt = validlabels.begin(); vIt != validlabels.end(); ++vIt){
		LabelPixelType labelValue = *vIt;

		if(labelValue != 0){
			// std::cout << "min: " << labelstatistics->GetMinimum( labelValue ) << std::endl;
			// std::cout << "max: " << labelstatistics->GetMaximum( labelValue ) << std::endl;
			// std::cout << "median: " << labelstatistics->GetMedian( labelValue ) << std::endl;
			// std::cout << "mean: " << labelstatistics->GetMean( labelValue ) << std::endl;
			// std::cout << "sigma: " << labelstatistics->GetSigma( labelValue ) << std::endl;
			// std::cout << "variance: " << labelstatistics->GetVariance( labelValue ) << std::endl;
			// std::cout << "sum: " << labelstatistics->GetSum( labelValue ) << std::endl;
			InputImageType::RegionType region = connectedregionsstatistics->GetRegion( labelValue );
			InputImageType::IndexType index = region.GetIndex();
			InputImageType::SizeType size = region.GetSize();
			cout << connectedregionsstatistics->GetCount( labelValue ) << ","<<size[0]<< ","<<size[1]<< ","<<size[2]<<endl;

			if(vectorimage){
				VectorImageIteratorType vectorit(radius, vectorimage, vectorimage->GetLargestPossibleRegion());
				VectorImageType::IndexType vectorimageindex;
				vectorimageindex[0] = index[0] + size[0]/2.0;
				vectorimageindex[1] = index[1] + size[1]/2.0;
				vectorimageindex[2] = index[2] + size[2]/2.0;
				vectorit.SetLocation(vectorimageindex);

				VectorImageType::Pointer outputimage = VectorImageType::New();

				VectorImageType::SizeType size;
				size[0] = radius[0]*2 + 1;
				size[1] = radius[1]*2 + 1;
				size[2] = radius[2]*2 + 1;
				VectorImageType::RegionType region;
				region.SetSize(size);

				outputimage->SetRegions(region);
				outputimage->SetVectorLength(vectorImageFilename.size());
				outputimage->Allocate();

				VectorImageIteratorType outit(radius, outputimage, outputimage->GetLargestPossibleRegion());
				outit.GoToBegin();

				int i = 0;


				while(!outit.IsAtEnd()){
					outit.SetCenterPixel(vectorit.GetPixel(i));				
					++outit;
					i++;
				}

				uuid_t id;
				uuid_generate(id);
			  	uuid_unparse(id, uuid);

	  		  	string outfilename = outputImageDirectory;
			  	outfilename.append(string(uuid)).append(".nrrd");

				cout<<"Writing file: "<<outfilename<<endl;
				VectorImageFileWriterType::Pointer writer = VectorImageFileWriterType::New();
				writer->SetFileName(outfilename);
				writer->SetInput(outputimage);
				writer->Update();	

			}
		}
		
	}

	delete[] uuid;

//                std::cout << "count: " << labelstatistics->GetCount( labelValue ) << std::endl;
//                std::cout << "region: " << labelstatistics->GetRegion( labelValue ) << std::endl;

	// VectorImageType::Pointer vectorimage = composeImageFilter->GetOutput();

	// VectorImageIteratorType::RadiusType radius;	
	// radius[0] = neighborhood[0];
	// radius[1] = neighborhood[1];
	// radius[2] = neighborhood[2];

	// VectorImageType::Pointer outputimage = VectorImageType::New();

	// VectorImageType::SizeType size;
	// size[0] = radius[0]*2 + 1;
	// size[1] = radius[1]*2 + 1;
	// size[2] = radius[2]*2 + 1;
	// VectorImageType::RegionType region;
	// region.SetSize(size);
	
	// outputimage->SetRegions(region);
	// outputimage->SetVectorLength(vectorImageFilename.size());
	// outputimage->Allocate();

	// InputRandomIteratorType randomit(labelimage, labelimage->GetLargestPossibleRegion());
	// randomit.SetNumberOfSamples(labelimage->GetLargestPossibleRegion().GetNumberOfPixels());
	// randomit.GoToBegin();

	// InputIteratorType init(radius, labelimage, labelimage->GetLargestPossibleRegion());
	

	// char *uuid = new char[100];	


	// VectorImageIteratorType vectorit(radius, vectorimage, vectorimage->GetLargestPossibleRegion());

	// cout<<"Searching for label: "<<labelValue<<endl;
	// if(labelValueContains != -1){
	// 	cout<<"The region contains label: "<<labelValueContains<<", ratio: "<<labelValueContainsPercentage<<endl;
	// }	
	
	// while(!randomit.IsAtEnd() && numberOfSamples){
	// 	if(randomit.Get() == labelValue && containsLabel(&init, randomit.GetIndex(), labelValueContains, labelValueContainsPercentage)){
	// 		uuid_t id;
	// 		uuid_generate(id);
	// 	  	uuid_unparse(id, uuid);

	// 	  	numberOfSamples--;

	// 	  	string outfilename = outputImageDirectory;
	// 	  	outfilename.append(string(uuid)).append(".nrrd");

	// 	  	// if(vectorimage->GetVectorLength() == 2){
	// 	  	// 	writeImage<2>(&vectorit, radius, outfilename);
	// 	  	// }else if(vectorimage->GetVectorLength() == 3){
	// 	  	// 	writeImage<3>(&vectorit, radius, outfilename);
	// 	  	// }else{
	// 	  	// 	throw "Modify the source code here and recompile to use with the appropriate number of components.";
	// 	  	// }

	// 		VectorImageIteratorType outit(radius, outputimage, outputimage->GetLargestPossibleRegion());
	// 		outit.GoToBegin();

	// 		vectorit.SetLocation(randomit.GetIndex());

	// 		int i = 0;
	// 		while(!outit.IsAtEnd()){
	// 			outit.SetCenterPixel(vectorit.GetPixel(i));				
	// 			++outit;
	// 			i++;
	// 		}

	// 		cout<<"Writing file: "<<outfilename<<endl;
	// 		VectorImageFileWriterType::Pointer writer = VectorImageFileWriterType::New();
	// 		writer->SetFileName(outfilename);
	// 		writer->SetInput(outputimage);
	// 		writer->Update();			
	// 	}
	// 	++init;
	// 	++vectorit;
	// 	++randomit;
	// }

	// delete[] uuid;


	return 0;
}