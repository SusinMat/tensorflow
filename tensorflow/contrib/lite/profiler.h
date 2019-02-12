#ifndef TENSORFLOW_CONTRIB_LITE_PROFILER_H__
#define TENSORFLOW_CONTRIB_LITE_PROFILER_H__

#include <chrono>
#include <list>
#include <iostream>
#include <string>

namespace tflite {

class Profiler {
public:
    /** Access the Profiler singleton
     *
     * @return The Profiler
     */
    static Profiler &get();

    void startInferenceMeasure();

    void endInferenceMeasure();

    void saveMeasure();

    void dumpMeasures(const std::string profiler_file_name);

    /** Prevent instances of this class from being copy constructed */
    Profiler(Profiler const&) = delete;

    /** Prevent instances of this class from being copied */
    void operator=(const Profiler &) = delete;

    /** Kernel information */
    struct KernelExecutionMeasure
    {
        /**< Kernel name */
        std::string name {};
        /**< Time it took the kernel to run */
        std::chrono::microseconds execution_time {};
        /**< Time in which the kernel start*/
        std::chrono::milliseconds::rep start_clock {};
        /**< Time in which the kernel stop */
        std::chrono::milliseconds::rep stop_clock {};
    };

    void addKernelMeasure(KernelExecutionMeasure measure);

private:
    /* Private constructor to prevent instancing. */
    Profiler();
    ~Profiler();

    std::list<KernelExecutionMeasure> _measures;
    std::chrono::time_point<std::chrono::system_clock> _start_execution_time;
    std::chrono::time_point<std::chrono::system_clock> _end_execution_time;
};

}

#endif
