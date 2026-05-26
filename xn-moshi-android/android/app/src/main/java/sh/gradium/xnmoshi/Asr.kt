package sh.gradium.xnmoshi

/**
 * Thin JNI wrapper around xn-moshi-android.
 *
 * One [Asr] handle = one streaming session. Calls are serialized inside Rust
 * with a [Mutex], but you should still keep them on a single background thread
 * to avoid wasting CPU on contention.
 *
 * Lifecycle: [init] → loop of [step] → [close]. Use [reset] to drop accumulated
 * tokens between utterances without recreating the session.
 */
class Asr private constructor(private val handle: Long) {

    companion object {
        init { System.loadLibrary("xn_moshi_android") }

        @JvmStatic external fun nativeInit(
            mimiPath: String, lmPath: String, tokenizerPath: String, dtype: String,
        ): Long

        @JvmStatic external fun nativeStep(handle: Long, pcm: FloatArray): String?
        @JvmStatic external fun nativeReset(handle: Long)
        @JvmStatic external fun nativeFrameSize(): Int
        @JvmStatic external fun nativeSampleRate(): Int
        @JvmStatic external fun nativeFree(handle: Long)

        val frameSize: Int get() = nativeFrameSize()
        val sampleRate: Int get() = nativeSampleRate()

        fun init(mimiPath: String, lmPath: String, tokenizerPath: String, dtype: String): Asr {
            val h = nativeInit(mimiPath, lmPath, tokenizerPath, dtype)
            check(h != 0L) { "nativeInit returned 0" }
            return Asr(h)
        }
    }

    /** Feed exactly [frameSize] PCM samples (mono, f32, [sampleRate] Hz). */
    fun step(pcm: FloatArray): String = nativeStep(handle, pcm) ?: ""

    fun reset() = nativeReset(handle)

    fun close() = nativeFree(handle)
}
