#include "createHistogramCLP.h"

#include <vtkActor.h>
#include <vtkBarChartActor.h>
#include <vtkFieldData.h>
#include <vtkImageAccumulate.h>
#include <vtkImageData.h>
#include <vtkImageExtractComponents.h>
#include <vtkIntArray.h>
#include <vtkJPEGReader.h>
#include <vtkLegendBoxActor.h>
#include <vtkProperty2D.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>
#include <vtkImageMagnitude.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageToHistogramFilter.h>

using namespace std;
 
int main (int argc, char * argv[]){

  PARSE_ARGS;


  if(inputFilename.compare("") == 0){
    cerr<<"Type "<<argv[0]<<" --help"<<endl;
    return EXIT_FAILURE;
  }

  typedef unsigned short InputPixelType; 
  typedef itk::Image<InputPixelType, 3> InputImageType;
  typedef itk::ImageFileReader<InputImageType>             InputImageReaderType;
 
  InputImageReaderType::Pointer reader = InputImageReaderType::New();
  reader->SetFileName(inputFilename);
  InputImageType::Pointer image = reader->GetOutput();

  typedef itk::Statistics::ImageToHistogramFilter< InputImageType > ImageToHistogramFilterType;
 
  ImageToHistogramFilterType::HistogramType::MeasurementVectorType lowerBound(255);
  lowerBound.Fill(0);
 
  ImageToHistogramFilterType::HistogramType::MeasurementVectorType upperBound(255);
  upperBound.Fill(255) ;
 
  ImageToHistogramFilterType::HistogramType::SizeType size(1);
  size.Fill(255);
 
  ImageToHistogramFilterType::Pointer imageToHistogramFilter = ImageToHistogramFilterType::New();
  imageToHistogramFilter->SetInput(image);
  imageToHistogramFilter->SetHistogramBinMinimum(lowerBound);
  imageToHistogramFilter->SetHistogramBinMaximum(upperBound);
  imageToHistogramFilter->SetHistogramSize(size);
  imageToHistogramFilter->Update();

  ImageToHistogramFilterType::HistogramType* histogram = imageToHistogramFilter->GetOutput();
  
 
  vtkSmartPointer<vtkIntArray> frequencies = vtkSmartPointer<vtkIntArray>::New();
  frequencies->SetNumberOfComponents(1);
  frequencies->SetNumberOfTuples(256);
 
  for(int j = 0; j < 256; ++j){
    frequencies->SetTuple1(j, histogram->GetFrequency(j));
  }
 
  vtkSmartPointer<vtkDataObject> dataObject = vtkSmartPointer<vtkDataObject>::New();
 
  dataObject->GetFieldData()->AddArray( frequencies );
 
  // Create a vtkBarChartActor
  vtkSmartPointer<vtkBarChartActor> barChart = vtkSmartPointer<vtkBarChartActor>::New();
 
  barChart->SetInput(dataObject);
  barChart->SetTitle("Histogram");
  barChart->GetPositionCoordinate()->SetValue(0.05,0.05,0.0);
  barChart->GetPosition2Coordinate()->SetValue(0.95,0.85,0.0);
  barChart->GetProperty()->SetColor(1,1,1);
 
  barChart->GetLegendActor()->SetNumberOfEntries(dataObject->GetFieldData()->GetArray(0)->GetNumberOfTuples());
  barChart->LegendVisibilityOff();
  barChart->LabelVisibilityOff();
  
  double red[3] = { 1, 0, 0 };
  int count = 0;
  for(int i = 0; i < 256; ++i){
    barChart->SetBarColor( count++, red );
  }
 
  // Visualize the histogram
  vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
  renderer->AddActor(barChart);
 
  vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->AddRenderer(renderer);
  renderWindow->SetSize(640, 480);
 
  vtkSmartPointer<vtkRenderWindowInteractor> interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
  interactor->SetRenderWindow(renderWindow);
  // Initialize the event loop and then start it
  interactor->Initialize();



  if(outputFilename.compare("") != 0){
    
    // Screenshot  
    vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter = vtkSmartPointer<vtkWindowToImageFilter>::New();
    windowToImageFilter->SetInput(renderWindow);
    windowToImageFilter->SetMagnification(3); //set the resolution of the output image (3 times the current resolution of vtk render window)
    windowToImageFilter->SetInputBufferTypeToRGBA(); //also record the alpha (transparency) channel
    //windowToImageFilter->ReadFrontBufferOff(); // read from the back buffer
    windowToImageFilter->Update();
    
    vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();
    writer->SetFileName(outputFilename.c_str());
    writer->SetInputConnection(windowToImageFilter->GetOutputPort());
    writer->Write();
  }
  
 
  if(!nogui){
    interactor->Start();
  }
 
  return  EXIT_SUCCESS;
}