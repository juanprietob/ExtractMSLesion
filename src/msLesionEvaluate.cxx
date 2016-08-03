#include "msLesionEvaluateCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkComposeImageFilter.h>
#include <itkVectorImage.h>
#include <itkImageFileWriter.h>
#include <itkNeighborhoodIterator.h>
#include <itkImageRegionIterator.h>
#include <itkFixedArray.h>
#include <itkVectorImageToImageAdaptor.h>
#include "itkEvaluateImage.h"
#include <itkImageToImageFilter.h>

#include <iostream>
#include <string>
#include <sstream>
#include <math.h>


#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>



using namespace std;

typedef unsigned short PixelType;
static const int Dimension = 3;

typedef itk::Image<PixelType, Dimension> InputImageType;
typedef InputImageType::IndexType InputImageIndexType;
typedef itk::ImageFileReader<InputImageType> InputImageFileReaderType;
typedef itk::ImageRegionIterator<InputImageType> InputImageRegionIteratorType;
typedef itk::ImageFileWriter<InputImageType> InputImageFileWriterType;

typedef unsigned short VectorImagePixelType;
typedef itk::VectorImage<VectorImagePixelType, Dimension> VectorImageType;  
typedef itk::ComposeImageFilter< InputImageType, VectorImageType> ComposeImageFilterType;
typedef itk::NeighborhoodIterator<VectorImageType> VectorImageIteratorType;
typedef VectorImageIteratorType::RadiusType VectorImageRadiusType;
typedef itk::ImageFileWriter<VectorImageType> VectorImageFileWriterType;
typedef itk::ImageRegionIterator<VectorImageType> VectorImageRegionIteratorType;


typedef double VectorImageDoublePixelType;
typedef itk::VectorImage<VectorImageDoublePixelType, Dimension> VectorImageDoubleType;  
typedef itk::ImageRegionIterator<VectorImageDoubleType> VectorImageDoubleRegionIteratorType;
typedef itk::ImageFileReader<VectorImageDoubleType> VectorImageDoubleFileReaderType;
typedef itk::VectorImageToImageAdaptor<VectorImageDoublePixelType, Dimension> ImageAdaptorType;

namespace itk{

    template<class TInputImage, class TOutput>
    class EvaluateImage: public ImageToImageFilter<TInputImage, TOutput>
    {
        public:

            typedef EvaluateImage           Self;
            typedef ImageToImageFilter<TInputImage, TOutput>     Superclass;
            typedef SmartPointer<Self>             Pointer;
            typedef SmartPointer<const Self>       ConstPointer;

            /** Run-time type information (and related methods). */
            itkTypeMacro(ImageToImageFilter, Superclass)

            /** Method for creation through the object factory. */
            itkNewMacro(Self)

            /** Some convenient typedefs. */
            typedef TInputImage                             InputImageType;
            typedef typename InputImageType::Pointer        InputImagePointer;
            typedef typename InputImageType::ConstPointer   InputImageConstPointer;
            typedef typename InputImageType::RegionType     InputImageRegionType;
            typedef typename InputImageType::PixelType      InputImagePixelType;
            typedef typename InputImageType::IndexType      IndexType;
            typedef typename TInputImage::PixelType         PixelType;
            typedef itk::NeighborhoodIterator<InputImageType> InputImageIteratorType;
            typedef typename InputImageIteratorType::RadiusType RadiusType;
            typedef itk::ImageFileWriter<InputImageType>    InputImageFileWriterType;



            /** Superclass typedefs. */
            typedef TOutput                                       OutputImageType;
            typedef typename OutputImageType::Pointer             OutputImagePointerType;
            typedef typename OutputImageType::RegionType          OutputImageRegionType;
            typedef typename OutputImageType::PixelType           OutputImagePixelType;
            typedef typename OutputImageType::IndexType           OutputImageIndexType;
            typedef itk::ImageRegionIterator<OutputImageType>     OutputImageIteratorType;

            /** Superclass typedefs. */
            typedef itk::Image< unsigned short, 3> InputImageLabelType;
            typedef typename InputImageLabelType::Pointer             InputImageLabelPointerType;
            typedef typename InputImageLabelType::RegionType          InputImageLabelegionType;
            typedef typename InputImageLabelType::PixelType           InputImageLabelPixelType;
            typedef typename InputImageLabelType::IndexType           InputImageLabelIndexType;
            typedef itk::ImageRegionIterator<InputImageLabelType>     InputImageLabelIteratorType;

            itkSetMacro(Radius, RadiusType)
            itkSetMacro(LabelImage, InputImageLabelPointerType)

        protected:

            EvaluateImage(){
                m_LabelImage = 0;
            }

            ~EvaluateImage(){
                
            }

            virtual void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE {

                //if(threadId == 1)
                {
                    InputImageType* inputimage = const_cast<InputImageType*>(this->GetInput());
                    OutputImagePointerType outputimage = this->GetOutput();

                    InputImageIteratorType init(m_Radius, inputimage, outputRegionForThread);
                    init.GoToBegin();

                    InputImageLabelIteratorType labelit;

                    if(m_LabelImage){
                        labelit = InputImageLabelIteratorType(m_LabelImage, outputRegionForThread);
                        labelit.GoToBegin();
                    }


                    InputImagePointer outimg = InputImageType::New();

                    VectorImageType::SizeType size;
                    size[0] = m_Radius[0]*2 + 1;
                    size[1] = m_Radius[1]*2 + 1;
                    size[2] = m_Radius[2]*2 + 1;
                    InputImageRegionType region;
                    region.SetSize(size);
                    
                    outimg->SetRegions(region);
                    outimg->SetVectorLength(inputimage->GetVectorLength());
                    outimg->Allocate();

                    InputImageIteratorType oit(m_Radius, outimg, outimg->GetLargestPossibleRegion());

                    OutputImageIteratorType outit(outputimage, outputRegionForThread);
                    outit.GoToBegin();

                    string outfilename = to_string(threadId) + ".nrrd";
                    //cout<<"Writing file: "<<outfilename<<endl;
                    typename InputImageFileWriterType::Pointer writer = InputImageFileWriterType::New();
                    writer->SetFileName(outfilename);

                    string command = "python /Users/prieto/NetBeansProjects/UNC/ExtractMSLesion/py/regularization.py --sample /Users/prieto/NetBeansProjects/UNC/ExtractMSLesion/bin/";
                    command+=outfilename;
                    command+=" --model /Volumes/BookStudio1/data/deepLearningExemplars1/deep1/regularization9.2.3.cpkt";

                    while(!init.IsAtEnd()){

                        bool evaluate = true;

                        if(m_LabelImage){
                            if(!(labelit.Get() == 6 || labelit.Get() == 8)){
                                evaluate = false;
                            }
                            ++labelit;
                        }

                        if(evaluate){
                            int i = 0;
                            oit.GoToBegin();
                            while(!oit.IsAtEnd()){
                                oit.SetCenterPixel(init.GetPixel(i));
                                ++oit;
                                i++;
                            }
                            
                            writer->SetInput(outimg);
                            writer->Update();

                            string res = exec(command.c_str());
                            float val = atof(res.c_str());
                            outit.Set(val*100);


                        }

                        ++init;
                        ++outit;
                    }
                }

            }
        private:
            RadiusType m_Radius;
            InputImageLabelPointerType m_LabelImage;

            std::string exec(const char* cmd) {
                char buffer[128];
                std::string result = "";
                std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
                if (!pipe) throw std::runtime_error("popen() failed!");
                while (!feof(pipe.get())) {
                    if (fgets(buffer, 128, pipe.get()) != NULL)
                        result += buffer;
                }
                return result;
            }
    };
}


int main (int argc, char * argv[]){


    PARSE_ARGS;

    if(vectorImageFilename.size() != 0){
        if(vectorImageFilename.size() == 0){
            cout<<"Type "<<argv[0]<<" --help to find out how to use this program."<<endl;
            return 1;
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
        VectorImageType::DirectionType direction;
        direction.SetIdentity();
        vectorimage->SetDirection(direction);

        InputImageType::Pointer labelImg = 0;

        if(labelMap.compare("") != 0){
            InputImageFileReaderType::Pointer readimage = InputImageFileReaderType::New();
            readimage->SetFileName(labelMap);
            readimage->Update();
            labelImg = readimage->GetOutput();
        }

        typedef itk::EvaluateImage<VectorImageType, InputImageType> EvaluateImageType;

        EvaluateImageType::Pointer evaluate = EvaluateImageType::New();

        EvaluateImageType::RadiusType radius;
        radius[0] = 3;
        radius[1] = 3;
        radius[2] = 1;

        evaluate->SetInput(vectorimage);
        evaluate->SetRadius(radius);
        if(labelImg){
            evaluate->SetLabelImage(labelImg);
        }
        evaluate->Update();
        InputImageType::Pointer output = evaluate->GetOutput();

        if(labelImg){
            output->SetDirection(labelImg->GetDirection());
            output->SetSpacing(labelImg->GetSpacing());
            output->SetOrigin(labelImg->GetOrigin());
        }

        InputImageFileWriterType::Pointer writer = InputImageFileWriterType::New();
        writer->SetInput(output);
        writer->SetFileName(outputLabelMap);
        writer->Update();
        
        // cout<<"Writing "<<outputLabelMap<<endl;
        // VectorImageFileWriterType::Pointer writer = VectorImageFileWriterType::New();
        // writer->SetFileName(outputLabelMap);
        // writer->SetInput(vectorimage);
        // writer->Update();
    }else if(probabilityMap.compare("") != 0){

        InputImageFileReaderType::Pointer readerlabel =  InputImageFileReaderType::New();
        readerlabel->SetFileName(labelMap);
        readerlabel->Update();

        InputImageType::Pointer inputlabelimg = readerlabel->GetOutput();


        VectorImageDoubleFileReaderType::Pointer reader = VectorImageDoubleFileReaderType::New();
        reader->SetFileName(probabilityMap);
        reader->Update();

        VectorImageDoubleType::Pointer vectorimage = reader->GetOutput();

        InputImageType::Pointer outlabelimg = InputImageType::New();
        outlabelimg->SetRegions(vectorimage->GetLargestPossibleRegion());
        outlabelimg->SetSpacing(inputlabelimg->GetSpacing());
        outlabelimg->SetOrigin(inputlabelimg->GetOrigin());
        outlabelimg->SetDirection(inputlabelimg->GetDirection());
        outlabelimg->Allocate();

        InputImageRegionIteratorType itoutlabel(outlabelimg, outlabelimg->GetLargestPossibleRegion());
        itoutlabel.GoToBegin();

        VectorImageDoubleRegionIteratorType vectit(vectorimage, vectorimage->GetLargestPossibleRegion());
        vectit.GoToBegin();

        InputImageRegionIteratorType itinlabel(inputlabelimg, inputlabelimg->GetLargestPossibleRegion());
        itinlabel.GoToBegin();

        
        


    }else{
        cout<<"type --help to check how to use this program"<<endl;
    }

    return EXIT_SUCCESS;



}