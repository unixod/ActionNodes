#ifndef ACTION_NODES_UTILS_THREAD_POOL_H
#define ACTION_NODES_UTILS_THREAD_POOL_H

#include <cassert>
#include <atomic>
#include <thread>
#include <vector>
#include <mutex>
#include <functional>
#include <any>
#include <condition_variable>

namespace anodes::utils {

class ThreadPool {
public:
    ThreadPool(std::size_t numOfThreads = std::thread::hardware_concurrency())
        : busyWorkerThreads_{numOfThreads}
    {
        if (numOfThreads == 0) {
            throw std::runtime_error{"A number of threads must be greater than 0."};
        }

        threads_.reserve(numOfThreads);
        for (auto i = numOfThreads; i; --i) {
            threads_.emplace_back( [this]{ runTh_(); });
        }
    }

    ThreadPool(ThreadPool&&) = delete;
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator = (ThreadPool&&) = delete;
    ThreadPool& operator = (const ThreadPool&) = delete;

    ~ThreadPool()
    {
        std::unique_lock lck{mx_};
        stop_ = true;
        lck.unlock();

        cvThs_.notify_all();

        for (auto& th : threads_) {
            th.join();
        }
    }

    template<
        typename MapCallable,
        typename ReduceCallable,
        typename RaIt,
        typename InitValue
    >
    auto mapReduce(RaIt begin, RaIt end, InitValue&& value, MapCallable&& mapFunc, ReduceCallable&& reduceFunc)
    {
        using RaItCategory = typename std::iterator_traits<RaIt>::iterator_category;
        static_assert(std::is_same_v<RaItCategory, std::random_access_iterator_tag>);

        auto procFunc = [&mapFunc, &reduceFunc, begin, this]() -> std::any {
            auto c = --cnt_;

            if (c > 0) {
                cvThs_.notify_one();
            }

            if (c < 0) {
                return {};
            }

            auto r = mapFunc(*std::next(begin, c));

            for (; c >= 0; c = --cnt_) {
                r = reduceFunc(r, mapFunc(*std::next(begin, c)));
            }

            return r;
        };

        using MapResultType = decltype(mapFunc(std::declval<typename std::iterator_traits<RaIt>::reference>()));
        using ReduceResultType = decltype(reduceFunc(std::declval<MapResultType>(), std::declval<MapResultType>()));

        ReduceResultType result{std::forward<InitValue>(value)};

        auto submitFunc = [&reduceFunc, &result](std::any& r){
            assert(r.has_value());
            auto& v = std::any_cast<ReduceResultType&>(r);
            result = reduceFunc(result, v);
        };

        runMain_(procFunc, submitFunc, std::distance(begin, end));
        return result;
    }

private:
    template<typename ProcCallable, typename SubmitCallable, typename Cnt>
    void runMain_(ProcCallable& procFunc, SubmitCallable& submitFunc, const Cnt cnt)
    {
        assert(cnt >= 0);
        assert(cnt <= std::numeric_limits<decltype(cnt_)::value_type>::max());
        proc_ = std::ref(procFunc);
        submit_ = std::ref(submitFunc);
        cnt_ = cnt;

        auto r = proc_();

        std::unique_lock lck{mx_};

        if (r.has_value()) {
            submit_(r);
        }

        cvMain_.wait(lck, [this]{
            return busyWorkerThreads_ == 0;
        });
    }

    void runTh_()
    {
        std::any r;

        while(!stop_) {
            std::unique_lock lck{mx_};

            if (r.has_value()) {
                submit_(r);
                r.reset();
            }

            assert(busyWorkerThreads_ > 0);
            if (--busyWorkerThreads_ == 0) {
                cvMain_.notify_one();
            }
            cvThs_.wait(lck, [this]{
                return cnt_ > 0 || stop_;
            });
            if (stop_) {
                return;
            }
            ++busyWorkerThreads_;
            lck.unlock();

            r = proc_();
        }
    }

private:
    bool stop_ = false;
    std::size_t busyWorkerThreads_;
    std::mutex mx_;
    std::vector<std::thread> threads_;
    std::condition_variable cvThs_;
    std::condition_variable cvMain_;

    std::function<std::any()> proc_;
    std::function<void(std::any&)> submit_;
    std::atomic<std::ptrdiff_t> cnt_ = 0;
};

} // namespace anodes::utils

#endif // ACTION_NODES_UTILS_THREAD_POOL_H
