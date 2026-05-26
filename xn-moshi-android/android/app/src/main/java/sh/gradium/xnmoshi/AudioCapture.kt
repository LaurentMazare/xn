package sh.gradium.xnmoshi

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log

/**
 * Mic capture sized to xn-moshi's 24 kHz / 1920-sample frame. Allocates one
 * recorder, exposes a blocking [read] that fills exactly one frame, and a
 * [close] that stops + releases.
 */
class AudioCapture(
    private val sampleRate: Int = Asr.sampleRate,
    private val frameSize: Int = Asr.frameSize,
) {
    private val minBuf = AudioRecord.getMinBufferSize(
        sampleRate,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_FLOAT,
    )

    @SuppressLint("MissingPermission") // caller checks RECORD_AUDIO
    private val recorder = AudioRecord.Builder()
        .setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)
        .setAudioFormat(
            AudioFormat.Builder()
                .setSampleRate(sampleRate)
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                .build(),
        )
        // 4 frames of headroom on top of the OS minimum, mostly to absorb GC.
        .setBufferSizeInBytes(maxOf(minBuf, frameSize * 4 * 4))
        .build()

    private val frame = FloatArray(frameSize)

    fun start() = recorder.startRecording()

    /** Fill one full frame; loops on partial reads. Returns null on error. */
    fun readFrame(): FloatArray? {
        var off = 0
        while (off < frameSize) {
            val n = recorder.read(frame, off, frameSize - off, AudioRecord.READ_BLOCKING)
            if (n < 0) {
                Log.w("AudioCapture", "AudioRecord.read=$n")
                return null
            }
            off += n
        }
        return frame
    }

    fun close() {
        try { recorder.stop() } catch (_: IllegalStateException) {}
        recorder.release()
    }
}
