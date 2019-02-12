#include "tensorflow/contrib/lite/profiler.h"
#include <fstream>
#include <iostream>
#include <chrono>

namespace tflite {

Profiler::Profiler()
    : _measures() {
    _start_execution_time = std::chrono::system_clock::now();
    _end_execution_time = std::chrono::system_clock::now();
}

Profiler::~Profiler(){
}

Profiler &Profiler::get(){
    static Profiler profiler;
    return profiler;
}

void Profiler::startInferenceMeasure(){
    _measures.clear();
    _start_execution_time = std::chrono::system_clock::now();
}

void Profiler::endInferenceMeasure() {
    _end_execution_time = std::chrono::system_clock::now();
}

void Profiler::dumpMeasures(const std::string profiler_file_name) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>
                            (_end_execution_time - _start_execution_time).count();

    std::ofstream output(profiler_file_name);
    if(output.fail()) {
        std::cout<< "[ERROR] Failed to create profiler dump file" << std::endl;
        return;
    }

    output << "{" << std::endl;
    output << "\t\"kernelMeasures\": [" << std::endl;
    int count(0);
    for(auto measure : _measures)
    {
        if (count++) {
            output << "," << std::endl;
        }
        output << "\t\t{" << std::endl;
        output << "\t\t\t\"kernel\": \"" << measure.name << "\"," << std::endl;
        output << "\t\t\t\"executionTime\": " << measure.execution_time.count() << "," << std::endl;
        output << "\t\t\t\"startExecutionTime\": " << measure.start_clock << "," << std::endl;
        output << "\t\t\t\"endExecutionTime\": " << measure.stop_clock << std::endl;

        output << "\t\t}";
    }
    output << std::endl;
    output << "\t]," << std::endl;
    output << "\t\"inferenceTime\": " << duration << "," << std::endl;
    output << "\t\"startInferenceTime\": "
           << std::chrono::duration_cast<std::chrono::milliseconds> (_start_execution_time.time_since_epoch()).count()
           << "," << std::endl;
    output << "\t\"endInferenceTime\": "
           << std::chrono::duration_cast<std::chrono::milliseconds> (_end_execution_time.time_since_epoch()).count() << std::endl;
    output << "}" << std::endl;

    output.close();
}

void Profiler::addKernelMeasure(KernelExecutionMeasure measure) {
    _measures.push_back(std::move(measure));
}

} // end namespace tflite
