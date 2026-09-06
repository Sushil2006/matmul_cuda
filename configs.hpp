#pragma once

#include <tuple>

template <int BM_, int BN_, int BK_>
struct SmemConfig
{
    static constexpr int BM = BM_;
    static constexpr int BN = BN_;
    static constexpr int BK = BK_;
};

template <int BM_, int BN_, int BK_, int TM_, int TN_>
struct BlockTiledConfig : SmemConfig<BM_, BN_, BK_>
{
    static constexpr int TM = TM_;
    static constexpr int TN = TN_;
};

template <typename TryLaunch, typename Head, typename... Tail>
bool dispatchConfig(std::tuple<Head, Tail...>, TryLaunch &tryLaunch)
{
    if (tryLaunch(Head{}))
        return true;

    if constexpr (sizeof...(Tail) == 0)
        return false;
    else
        return dispatchConfig(std::tuple<Tail...>{}, tryLaunch);
}
