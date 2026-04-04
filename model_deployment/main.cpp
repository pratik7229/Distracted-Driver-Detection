#include<iostream>
#include<thread>
#include <opencv2/opencv.hpp>
#include<vector>
#include<string>
#include <onnxruntime/onnxruntime_cxx_api.h>

class Inference{
    private:
        std::vector<std::string> classes = {
            "normal driving",
            "texting right",
            "phone right",
            "texting left",
            "phone left",
            "radio",
            "drinking",
            "reaching behind",
            "hair/makeup",
            "talking passenger"
        };
        const int IMAGE_WIDTH = 224;
        const int IMAGE_LENGTH = 224;
        std::string MODEL_PATH = "/Users/pratik/Documents/Finalized Projects/Distracted_Driver_detection/model_training/model.onnx";
        Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "DriverMonitor"};
        Ort::SessionOptions session_options;
        std::unique_ptr<Ort::Session> session;

        std::vector<std::string> input_name_storage;
        std::vector<std::string> output_name_storage;

        std::vector<const char*> input_names;
        std::vector<const char*> output_names;
    public:
        Inference() {
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            session = std::make_unique<Ort::Session>(
                env,
                MODEL_PATH.c_str(),
                session_options
            );
            Ort::AllocatorWithDefaultOptions allocator;

            // Store as std::string (safe ownership)
            std::string input_name_str = session->GetInputNameAllocated(0, allocator).get();
            std::string output_name_str = session->GetOutputNameAllocated(0, allocator).get();

            // Store internally (class members)
            input_name_storage.push_back(input_name_str);
            output_name_storage.push_back(output_name_str);

            // Now store stable C-style pointers
            input_names.push_back(input_name_storage[0].c_str());
            output_names.push_back(output_name_storage[0].c_str());

            std::cout << "Input name: " << input_names[0] << std::endl;
            std::cout << "Output name: " << output_names[0] << std::endl;
        }
        // here the return type will be cv::mat 
        cv::Mat preprocessing(const cv::Mat& frame){
            cv::Mat resize, float_img;
            cv::resize(frame, resize, cv::Size(IMAGE_WIDTH, IMAGE_LENGTH));
            resize.convertTo(float_img, CV_32F, 1.0/255.0);

            return float_img;
        }

        int inference_output(const cv::Mat& frame) {

            // Convert cv::Mat → vector<float>
            std::vector<float> input_tensor_values(IMAGE_WIDTH * IMAGE_LENGTH * 3);

            std::memcpy(input_tensor_values.data(), frame.data,
                        input_tensor_values.size() * sizeof(float));

            std::vector<int64_t> input_shape = {1, IMAGE_LENGTH, IMAGE_WIDTH, 3};

            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                input_tensor_values.data(),
                input_tensor_values.size(),
                input_shape.data(),
                input_shape.size()
            );

            auto output_tensors = session->Run(
                Ort::RunOptions{nullptr},
                input_names.data(),
                &input_tensor,
                1,
                output_names.data(),
                1
            );

            // Get output
            float* output_data = output_tensors[0].GetTensorMutableData<float>();

            // Find max probability index
            int max_index = 0;
            float max_value = output_data[0];

            for (int i = 1; i < 10; i++) {
                if (output_data[i] > max_value) {
                    max_value = output_data[i];
                    max_index = i;
                }
            }

            return max_index;
        }

        void run(){
            cv::VideoCapture cap(0);
            if(!cap.isOpened()){
                std::cout<<"Failed to open the camera feed";
            }
            while(true){
                cv::Mat frame;
                cap >> frame;

                if(frame.empty()){
                    std::cerr << "Empty frame \n";
                    break;
                }

                // preprocessing
                cv::Mat processedFrame = preprocessing(frame);

                // inference
                int class_found = inference_output(processedFrame);

                // display the class

                std::string label = classes[class_found];

                cv::putText(frame, label, {20,40}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 255, 0}, 2);
                std::cout << "Prediction: " << label << std::endl;

                cv::imshow("Driver Feed", frame);

                if(cv::waitKey(1) == 27) break;
            }

        }



};
int main(){

    Inference inf;
    inf.run();

    return 0;
}