package sh.gradium.ptts

/**
 * Thin JNI wrapper around the pocket-tts Rust library.
 *
 * Usage: [init] → loop of ([nextChunk] until it returns null) → read [stats]
 * → [free]. Keep calls on a single background thread; the Rust session
 * holds an internal Mutex but we don't want contention at the JNI boundary.
 */
class Ptts private constructor(private val handle: Long) {

    companion object {
        init { System.loadLibrary("ptts_android") }

        @JvmStatic external fun nativeInit(weightsDir: String, voice: String): Long

        @JvmStatic external fun nativeGenerate(
            handle: Long, text: String, temperature: Float, seed: Long
        )

        @JvmStatic external fun nativeNextChunk(handle: Long): FloatArray?
        @JvmStatic external fun nativeStats(handle: Long): FloatArray
        @JvmStatic external fun nativeSampleRate(handle: Long): Int
        @JvmStatic external fun nativeSetThreads(handle: Long, n: Int)
        @JvmStatic external fun nativeNumCpus(): Int
        @JvmStatic external fun nativeFree(handle: Long)

        fun numCpus(): Int = nativeNumCpus()

        fun init(weightsDir: String, voice: String): Ptts {
            val h = nativeInit(weightsDir, voice)
            check(h != 0L) { "nativeInit returned 0" }
            return Ptts(h)
        }
    }

    fun generate(text: String, temperature: Float, seed: Long) =
        nativeGenerate(handle, text, temperature, seed)

    fun nextChunk(): FloatArray? = nativeNextChunk(handle)

    fun sampleRate(): Int = nativeSampleRate(handle)

    fun setThreads(n: Int) = nativeSetThreads(handle, n)

    fun stats(): Stats {
        val s = nativeStats(handle)
        return Stats(
            rtf = s[0], avgStepMs = s[1], totalElapsedS = s[2],
            durationS = s[3], firstAudioS = s[4], peakRssMb = s[5],
        )
    }

    fun close() = nativeFree(handle)

    data class Stats(
        val rtf: Float,
        val avgStepMs: Float,
        val totalElapsedS: Float,
        val durationS: Float,
        val firstAudioS: Float,
        val peakRssMb: Float,
    )
}
