#include <cstdint>

#include "Wrapper.h"
#include "ScreamTx.h"
#include "RtpQueue.h"

#include <sys/time.h>
#include <unistd.h>

#define V2

void packet_free(void *buf, uint32_t ssrc)
{
	free(buf);
}

namespace video
{
    extern int getEncoderRate();
}

namespace scream
{

bool useL4S = true;

float qdelay = 0.2f;

std::mutex _lock;
std::mutex lock_rtp_queue;

std::thread transmitThread;
bool transmitThreadRunning = false;

std::thread logThread;
bool stopLogThread = false;
bool logThreadRunning = false;

// Accumulated pace time, used to avoid starting very short pace timers
//  this can save some complexity at very higfh bitrates
float accumulatedPaceTime = 0.0f;

float minPaceInterval = 0.001f;
int minPaceIntervalUs = 900;

bool sendingPacket = false;

uint32_t rtcp_rx_time_ntp = 0;
double t0 = 0;

int fixedRate = 0;
bool disablePacing = 0;
int initRate = 700;
int minRate = 700;

//int maxRate = 20000;
int maxRate = 8000;

bool enableClockDriftCompensation = false;
#ifdef V2
float packetPacingHeadroom = 1.5f;
float scaleFactor = 0.7f;
ScreamV2Tx *screamTx = 0;
float bytesInFlightHeadroom = 2.0f;
float multiplicativeIncreaseFactor = 0.05f;
#else
int rateIncrease = 10000;
float rateScale = 0.5f;
float dscale = 10.0f;
float packetPacingHeadroom = 1.25f;
float scaleFactor = 0.9f;
ScreamV1Tx *screamTx = 0;
float txQueueSizeFactor = 0.1f;
float queueDelayGuard = 0.05f;
float fastIncreaseFactor = 1.0f;
bool isNewCc = false;
#endif

float postCongestionDelay = 4.0f;
float adaptivePaceHeadroom = 1.5f;
float hysteresis = 0.0f;

int periodicRateDropInterval = 600; // seconds*10

uint32_t lastKeyFrameT_ntp = 0;
int mtu = 1200;
float runTime = -1.0;
bool stopThread = false;

float delayTarget = 0.06f;
bool printSummary = true;

RtpQueue *rtpQueue = 0;

uint32_t getTimeInNtp()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    double time = tp.tv_sec + tp.tv_usec*1e-6 - t0;
    uint64_t ntp64 = uint64_t(time*65536.0);
    uint32_t ntp = 0xFFFFFFFF & ntp64;
    return ntp;
}

void Init()
{
#ifdef V2
	screamTx = new ScreamV2Tx(
		scaleFactor,
		scaleFactor,
		delayTarget,
		(initRate * 100) / 8,
		packetPacingHeadroom,
		adaptivePaceHeadroom,
		bytesInFlightHeadroom,
		multiplicativeIncreaseFactor,
		useL4S,
		false,
		false,
		enableClockDriftCompensation);
#else
	screamTx = new ScreamV1Tx(scaleFactor, scaleFactor,
		delayTarget,
		false,
		1.0f, dscale,
		(initRate * 100) / 8,
		packetPacingHeadroom,
		20,
		useL4S,
		false,
		enableClockDriftCompensation,
	    1.0f,
	    isNewCc);
	screamTx->setFastIncreaseFactor(fastIncreaseFactor);
#endif

	rtpQueue = new RtpQueue();

	screamTx->setCwndMinLow(5000);
    //screamTx->setPostCongestionDelay(postCongestionDelay);
    if (disablePacing)
		screamTx->enablePacketPacing(false);
}

void RegisterNewStream(uint32_t ssrc)
{
	printf("SCREAM: RegisterNewStream(%u)\n", ssrc);
#ifdef V2
        screamTx->registerNewStream(rtpQueue,
            ssrc,
            1.0f,
            minRate * 1000,
            initRate * 1000,
            maxRate * 1000,
            qdelay, // qdelay
            false,
            hysteresis);
#else
        screamTx->registerNewStream(rtpQueue,
            ssrc,
            1.0f,
            minRate * 1000,
            initRate * 1000,
            maxRate * 1000,
            rateIncrease * 1000,
            rateScale,
            qdelay, // qdelay
            txQueueSizeFactor,
            queueDelayGuard,
            scaleFactor,
            scaleFactor,
            false,
            hysteresis);
#endif
}

void sendPacket(const boost::asio::ip::udp::endpoint &peer, boost::asio::ip::udp::socket &sock, void* buf, int size)
{
	if (!buf) return;
    sendingPacket = true;
	try
	{
		sock.send_to(boost::asio::buffer(buf, size), peer);
	}
	catch (std::exception const& ex)
	{
		printf("SCREAM: sendPacket exception:%s\n", ex.what());
	}
    sendingPacket = false;
}

void NewMediaFrame(uint32_t ts, uint32_t ssrc, uint8_t *buf, int size, uint16_t seqNr, bool isMark)
{
	if (!screamTx) return;

	//printf("SCREAM: NewMediaFrame(sz:%d mark:%d)\n", size, isMark);

	ts = getTimeInNtp();

	// Store a copy of the buffer
	uint8_t *cpy = (uint8_t *)malloc(size);
	if (!cpy)
	{
		printf("SCREAM: OOM sz:%d\n", size);
		return;
	}
	memcpy(cpy, buf, size);

	{
		std::lock_guard lock { lock_rtp_queue };
		if (!rtpQueue->push(cpy, size, ssrc, seqNr, isMark, ts / 65536.0f))
		{
			printf("SCREAM: NewMediaFrame: RTPQUEUE IS FULL! sz:%d\n", rtpQueue->sizeOfQueue());
			delete cpy;
			return;
		}
	}
	
	screamTx->newMediaFrame(ts, ssrc, size, isMark);
}

void ProcessRTCP(unsigned char *buf_rtcp, int size)
{
	//printf("SCREAM: ProcessRTCP(sz:%d)\n", size);

#if 0
    std::string str;
    for (int i = 0; i < std::min(size, 15); ++i)
            str += std::to_string(int(buf_rtcp[i])) + ' ';
	printf("SCREAM: RTCP: '%s'\n", str.c_str());
#endif

	if (!screamTx) return;

	uint32_t time_ntp = getTimeInNtp(); // We need time in microseconds, roughly ms granularity is OK
	char s[100];
	const bool ntp = false;
	if (ntp) {
		struct timeval tp;
		gettimeofday(&tp, NULL);
		double time = tp.tv_sec + tp.tv_usec*1e-6;
		sprintf(s, "%1.6f", time);
	}
	else {
		sprintf(s, "%1.4f", time_ntp / 65536.0f);
	}
	screamTx->setTimeString(s);

	screamTx->incomingStandardizedFeedback(time_ntp, buf_rtcp, size);

	rtcp_rx_time_ntp = time_ntp;
}

uint32_t lastLogT_ntp = 0;
void logThreadProc()
{
    logThreadRunning = true;

    while (!stopLogThread)
    {
		uint32_t time_ntp = getTimeInNtp();
		bool isFeedback = time_ntp - rtcp_rx_time_ntp < 65536; // 1s in Q16
		if ((printSummary || !isFeedback) && time_ntp - lastLogT_ntp > 2 * 65536) // 2s in Q16
		{
			if (!isFeedback)
            {
                std::cerr << "SCREAM: No RTCP feedback received" << endl;
			}
			else if (screamTx)
            {
                std::lock_guard lock { _lock };

                int targetRate = screamTx->getTargetBitrate(VIDEO_SSRC) / 1000.0f;
                int transmitRate = screamTx->statistics->getAvgTransmitRate() / 1000.0f;

                float time_s = time_ntp / 65536.0f;
                char s[500];
                screamTx->getStatistics(time_s, s);

                int diff = transmitRate - targetRate;
                std::cout << "SCREAM: " << s << ", Target rate = " << targetRate << "kbps, diff = " << diff << "kbps" << std::endl;
			}
			lastLogT_ntp = time_ntp;
		}

        usleep(50000);
    }
}

/*
 * Transmit a packet if possible.
 * If not allowed due to packet pacing restrictions,
 * then start a timer.
 */
void transmitRtpThread(boost::asio::ip::udp::socket &sock, const boost::asio::ip::udp::endpoint &peer)
{
	int size;
	uint16_t seqNr;
    bool isMark;
	uint32_t time_ntp = getTimeInNtp();
	int sleepTime_us = 10;
	float retVal = 0.0f;
	int sizeOfQueue;
	struct timeval start, end;
	useconds_t diff = 0;
	float paceIntervalFixedRate = 0.0f;

	transmitThreadRunning = true;

	if (fixedRate > 0 && !disablePacing)
	{
		paceIntervalFixedRate = (mtu + 40)*8.0f / (fixedRate * 1000)*0.9;
	}
	for (;;)
	{
		if (stopThread)
		{
			return;
		}

		sleepTime_us = 10;
		retVal = 0.0f;
		uint32_t ssrc = 0;
#if 1
		{
			std::lock_guard lock { _lock };
			time_ntp = getTimeInNtp();
			retVal = screamTx->isOkToTransmit(time_ntp, ssrc);
		}

		if (retVal != -1.0f)
#endif
		{
			{
				std::lock_guard lock { lock_rtp_queue };
				sizeOfQueue = rtpQueue->sizeOfQueue();
			}
			do
			{
				gettimeofday(&start, 0);
				time_ntp = getTimeInNtp();

				retVal = screamTx->isOkToTransmit(time_ntp, ssrc);
				if (fixedRate > 0 && retVal >= 0.0f && sizeOfQueue > 0)
					retVal = paceIntervalFixedRate;
				if (disablePacing && sizeOfQueue > 0 && retVal > 0.0f)
					retVal = 0.0f;
				if (retVal > 0.0f)
					accumulatedPaceTime += retVal;
				if (retVal != -1.0 && ssrc == VIDEO_SSRC)
				{
                    void *buf;
                    uint32_t ssrc_unused;
					{
						std::lock_guard lock { lock_rtp_queue };
						rtpQueue->pop(&buf, size, ssrc_unused, seqNr, isMark);
                        if (ssrc_unused != VIDEO_SSRC)
                        {
                            printf("SCREAM: invalid SSRC %d used!", ssrc_unused);
                        }
						sendPacket(peer, sock, buf, size);
						//printf("SCREAM: sendPacket(%u buf:%p sz:%d)\n", ssrc, buf, size);
					}
                    packet_free(buf, ssrc);
                    buf = NULL;
					{
						std::lock_guard lock { _lock };
						time_ntp = getTimeInNtp();
						retVal = screamTx->addTransmitted(time_ntp, ssrc, size, seqNr, isMark);
						//printf("SCREAM: addTransmitted(%u seq:%d): paceInterval:%f\n", ssrc, int(seqNr), retVal);
					}
				}

				{
					std::lock_guard lock { lock_rtp_queue };
					sizeOfQueue = rtpQueue->sizeOfQueue();
				}
				gettimeofday(&end, 0);
				diff = end.tv_usec - start.tv_usec;
				accumulatedPaceTime = std::max(0.0f, accumulatedPaceTime - diff * 1e-6f);
			} while (accumulatedPaceTime <= minPaceInterval &&
				retVal != -1.0f &&
				sizeOfQueue > 0 &&
				!stopThread);
			if (accumulatedPaceTime > 0)
			{
				sleepTime_us = std::min((int)(accumulatedPaceTime*1e6f), minPaceIntervalUs);
				accumulatedPaceTime = 0.0f;
			}
		}
		usleep(sleepTime_us);
		sleepTime_us = 0;
	}
}

float AddTransmitted(uint32_t ts, // Wall clock ts when packet is transmitted
            uint32_t ssrc,
            int size,
            uint16_t seqNr,
            bool isMark)
{
	printf("SCREAM: AddTransmitted(ts:%u ssrc:%u sz:%d seq:%u marker:%d)\n", ts, ssrc, size, seqNr, isMark);

	if (!screamTx) return -1.0f;
	ts = getTimeInNtp();
	return screamTx->addTransmitted(ts,
            ssrc,
            size,
            seqNr,
            isMark);
}

float GetTargetBitrate(uint32_t ssrc)
{
	if (!screamTx) return -1.0f;
	return screamTx->getTargetBitrate(ssrc);
}

void GetStatistics(float time, char *s, size_t size)
{
	if (!screamTx) return;
	screamTx->getStatistics(time, s);
}

std::mutex &GetLock()
{
	return _lock;
}

bool IsLossEpoch(uint32_t ssrc)
{
	if (!screamTx) return false;
	bool ret = screamTx->isLossEpoch(ssrc);
	//if (ret) printf("SCREAM: IsLossEpoch: scream IDR. qsize:%d\n", rtpQueue->sizeOfQueue());
	return ret;
}

void StartStreaming(uint32_t ssrc, const boost::asio::ip::udp::endpoint &peer, boost::asio::ip::udp::socket &sock)
{
	printf("SCREAM: StartStreaming(ssrc:%u)\n", ssrc);

	struct timeval tp;
	gettimeofday(&tp, NULL);
	t0 = tp.tv_sec + tp.tv_usec*1e-6 - 1e-3;

	stopThread = false;
	transmitThread = std::thread { transmitRtpThread, std::ref(sock), peer };

#if 0
    stopLogThread = false;
	logThread = std::thread { logThreadProc };
#endif
}

void StopStreaming(uint32_t ssrc)
{
	printf("SCREAM: StopStreaming(%u)\n", ssrc);
	if (transmitThreadRunning)
	{
		stopThread = true;

		printf("SCREAM: StopStreaming(%u) waiting for transmitRtpThread to stop ... sendingPacket:%d\n", ssrc, sendingPacket);
		transmitThread.join();
		printf("SCREAM: StopStreaming(%u) transmitRtpThread DONE.\n", ssrc);
		transmitThreadRunning = false;
	}

	if (logThreadRunning)
	{
		stopLogThread = true;

		printf("SCREAM: StopStreaming(%u) waiting for logThread to stop ...\n", ssrc);
		logThread.join();
		printf("SCREAM: StopStreaming(%u) logThread DONE.\n", ssrc);
		logThreadRunning = false;
	}
}

bool SetECT(int sock, int value, bool ipv6)
{
    int iptos = 0;

    // Get current TOS value
    socklen_t toslen = sizeof(iptos);
    int retVal = getsockopt(sock, ipv6 ? IPPROTO_IPV6 : IPPROTO_IP, ipv6 ? IPV6_TCLASS : IP_TOS,  &iptos, &toslen);
    if (retVal < 0)
    {
        printf("ECN: ERR: Failed to get TOS marking on socket %d. err:%d\n", sock, retVal);
        iptos = 0;
    }
    else
    {
        printf("ECN: Got TOS %d before setting ECN. toslen:%u retVal:%d\n", iptos, toslen, retVal);
    }

    // Set ECT on the last two bits
    iptos = (iptos & 0xFC) | value;

    printf("ECN: Setting tos to %d ipv6:%d\n", iptos, ipv6);
    retVal = setsockopt(sock, ipv6 ? IPPROTO_IPV6 : IPPROTO_IP, ipv6 ? IPV6_TCLASS : IP_TOS, &iptos, sizeof(iptos));
    if (retVal < 0)
    {
        printf("ECN: ERR: Not possible to set ECN bits. retVal:%d \n", retVal);
        return false;
    }

    return true;
}


} // namespace scream
