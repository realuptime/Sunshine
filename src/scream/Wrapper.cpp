#include <cstdint>

#include "Wrapper.h"
#include "ScreamTx.h"
#include "RtpQueue.h"

#include <sys/time.h>

#define V2

void packet_free(void *buf, uint32_t ssrc)
{
	free(buf);
}

namespace scream {

/*
* ECN capable
* -1 = Not-ECT
* 0 = ECT(0)
* 1 = ECT(1)
* 3 = CE
*/
int ect = -1;

uint32_t rtcp_rx_time_ntp = 0;
double t0 = 0;
bool ntp = false; // use NTP timestamp in logfile

float FPS = 60.0f; // Frames per second
int fixedRate = 0;
bool isKeyFrame = false;
bool disablePacing = false;
float keyFrameInterval = 0.0;
float keyFrameSize = 1.0;
int initRate = 1000;
int minRate = 1000;
int maxRate = 200000;
bool enableClockDriftCompensation = false;
float burstTime = -1.0;
float burstSleep = -1.0;
bool isBurst = false;
float burstStartTime = -1.0;
float burstSleepTime = -1.0;
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
float bytesInFlightHeadroom = 1.25f;
float txQueueSizeFactor = 0.1f;
float queueDelayGuard = 0.05f;
float fastIncreaseFactor = 1.0f;
bool isNewCc = false;
#endif

float postCongestionDelay = 4.0f;
float adaptivePaceHeadroom = 1.5f;
float hysteresis = 0.0f;

int periodicRateDropInterval = 600; // seconds*10


uint16_t seqNr = 0;
uint32_t lastKeyFrameT_ntp = 0;
int mtu = 1200;
float runTime = -1.0;
bool stopThread = false;
pthread_t create_rtp_thread = 0;
pthread_t transmit_rtp_thread = 0;
pthread_t rtcp_thread = 0;
pthread_t sierra_python_thread = 0;
bool sierraLog;

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
                ect == 1,
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
                ect == 1,
                false,
                enableClockDriftCompensation,
              2.0f,
              isNewCc);
            screamTx->setFastIncreaseFactor(fastIncreaseFactor);
#endif

	rtpQueue = new RtpQueue();

	screamTx->setCwndMinLow(5000);
    screamTx->setPostCongestionDelay(postCongestionDelay);
    if (disablePacing)
		screamTx->enablePacketPacing(false);
}

void RegisterNewStream(uint32_t ssrc)
{
#ifdef V2
        screamTx->registerNewStream(rtpQueue,
            ssrc,
            1.0f,
            minRate * 1000,
            initRate * 1000,
            maxRate * 1000,
            0.2f,
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
            0.2f,
            txQueueSizeFactor,
            queueDelayGuard,
            scaleFactor,
            scaleFactor,
            false,
            hysteresis);
#endif
}

void NewMediaFrame(uint32_t time_ntp, uint32_t ssrc, int bytesRtp, bool isMarker)
{
	screamTx->newMediaFrame(time_ntp, ssrc, bytesRtp, isMarker);
}

void ProcessRTCP(unsigned char *buf_rtcp, int size)
{
	uint32_t time_ntp = getTimeInNtp(); // We need time in microseconds, roughly ms granularity is OK
	char s[100];
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

float IsOkToTransmit(uint32_t time_ntp, uint32_t &ssrc)
{
	return screamTx->isOkToTransmit(time_ntp, ssrc);
}

float AddTransmitted(uint32_t timestamp_ntp, // Wall clock ts when packet is transmitted
            uint32_t ssrc,
            int size,
            uint16_t seqNr,
            bool isMark)
{
	return screamTx->addTransmitted(timestamp_ntp,
            ssrc,
            size,
            seqNr,
            isMark);
}

float GetTargetBitrate(uint32_t ssrc)
{
	return screamTx->getTargetBitrate(ssrc);
}

void GetStatistics(float time, char *s, size_t size)
{
	screamTx->getStatistics(time, s);
}


} // namespace
