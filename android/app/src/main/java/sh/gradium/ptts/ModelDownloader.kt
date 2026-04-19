package sh.gradium.ptts

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Streams the pocket-tts weights + tokenizer + voice embeddings from
 * huggingface.co/kyutai/pocket-tts into the app's private files dir on first
 * launch.
 *
 * The caller is expected to use [ensureDownloaded] inside a coroutine tied to
 * the activity lifecycle. Progress is surfaced via [progress].
 */
class ModelDownloader(private val ctx: Context) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.MINUTES)
        .build()

    private val _progress = MutableStateFlow(Progress("idle", 0, 0))
    val progress: StateFlow<Progress> = _progress.asStateFlow()

    val root: File get() = File(ctx.filesDir, "models")

    data class Progress(val label: String, val received: Long, val total: Long)

    companion object {
        // Use the no-auth public variant. The "embeddings_v2/" files ship as
        // pre-primed KV-cache state; the Rust core loads them via
        // load_voice_kv_cache (see pocket-tts-android/src/lib.rs) rather than
        // via prompt_audio.
        private const val BASE =
            "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/main"
        val VOICES = listOf(
            "alba", "marius", "javert", "fantine", "cosette", "eponine", "azelma",
        )
    }

    /** Returns the root directory passed to [Ptts.init]. */
    suspend fun ensureDownloaded(): File = withContext(Dispatchers.IO) {
        val root = this@ModelDownloader.root
        root.mkdirs()

File(root, "embeddings_v2").mkdirs()
        fetch("$BASE/tts_b6369a24.safetensors", File(root, "tts_b6369a24.safetensors"), "model")
        fetch("$BASE/tokenizer.model", File(root, "tokenizer.model"), "tokenizer")
        for (v in VOICES) {
            fetch(
                "$BASE/embeddings_v2/$v.safetensors",
                File(root, "embeddings_v2/$v.safetensors"),
                "voice $v",
            )
        }
        _progress.value = Progress("done", 0, 0)
        root
    }

    private fun fetch(url: String, dest: File, label: String) {
        if (dest.exists() && dest.length() > 0) return
        _progress.value = Progress(label, 0, 0)
        val req = Request.Builder().url(url).build()
        client.newCall(req).execute().use { resp ->
            check(resp.isSuccessful) { "HTTP ${resp.code} for $url" }
            val body = resp.body ?: error("empty body for $url")
            val total = body.contentLength()
            val tmp = File(dest.parentFile, dest.name + ".part")
            body.byteStream().use { input ->
                tmp.outputStream().use { out ->
                    val buf = ByteArray(128 * 1024)
                    var received = 0L
                    while (true) {
                        val n = input.read(buf)
                        if (n <= 0) break
                        out.write(buf, 0, n)
                        received += n
                        _progress.value = Progress(label, received, total)
                    }
                }
            }
            check(tmp.renameTo(dest)) { "rename ${tmp} -> ${dest} failed" }
        }
    }
}
