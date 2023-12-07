#pragma once

#include <mutex>
#include <boost/asio.hpp>

#define VIDEO_SSRC 1

namespace scream
{
void Init();
void RegisterNewStream(uint32_t ssrc);
void NewMediaFrame(uint32_t ts, uint32_t ssrc, uint8_t *buf, int size, uint16_t seqNr, bool isMark);
void ProcessRTCP(unsigned char *buf_rtcp, int size);
float AddTransmitted(uint32_t ts, // Wall clock ts when packet is transmitted
            uint32_t ssrc,
            int size,
            uint16_t seqNr,
            bool isMark);
float GetTargetBitrate(uint32_t ssrc);
void GetStatistics(float time, char *s, size_t size);
std::mutex &GetLock();
bool IsLossEpoch(uint32_t ssrc);
void StartStreaming(uint32_t ssrc, const boost::asio::ip::udp::endpoint &peer, boost::asio::ip::udp::socket &sock);
void StopStreaming(uint32_t ssrc);
bool SetECT(int sock, int value, bool ipv6);
void SetMaxBitrate(int value);
} // namespace scream
