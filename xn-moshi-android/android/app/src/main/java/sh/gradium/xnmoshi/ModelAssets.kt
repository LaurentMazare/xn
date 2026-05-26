package sh.gradium.xnmoshi

import android.content.Context
import android.util.Log
import java.io.File

/**
 * Copies the three ASR model files from APK assets to the app's private
 * files dir once, then hands back absolute paths so the Rust side can mmap
 * them via safetensors.
 *
 * Drop the three files into `app/src/main/assets/`:
 *  - `mimi.safetensors`     (HF `kyutai/stt-2.6b-en-candle` → mimi-pytorch-e351c8d8@125.safetensors)
 *  - `lm.safetensors`       (same repo → model.safetensors)
 *  - `tokenizer.model`      (same repo → tokenizer_en_audio_4000.model)
 */
object ModelAssets {

    data class Paths(val mimi: String, val lm: String, val tokenizer: String)

    private const val TAG = "ModelAssets"
    private val FILES = listOf("mimi.safetensors", "lm.safetensors", "tokenizer.model")

    fun extract(ctx: Context): Paths {
        val dir = File(ctx.filesDir, "models").apply { mkdirs() }
        for (name in FILES) {
            val out = File(dir, name)
            if (out.exists() && out.length() > 0) continue
            Log.i(TAG, "extracting $name…")
            ctx.assets.open(name).use { input ->
                out.outputStream().use { output -> input.copyTo(output, bufferSize = 1 shl 20) }
            }
        }
        return Paths(
            mimi = File(dir, "mimi.safetensors").absolutePath,
            lm = File(dir, "lm.safetensors").absolutePath,
            tokenizer = File(dir, "tokenizer.model").absolutePath,
        )
    }
}
