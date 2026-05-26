package sh.gradium.xnmoshi

import android.content.Context
import android.util.Log
import java.io.File

/**
 * Copies the four ASR model files from APK assets to the app's private
 * files dir on first launch, then hands back absolute paths so the Rust
 * side can mmap them.
 *
 * The four files come from `...` on Hugging Face and
 * are bundled verbatim under `app/src/main/assets/`:
 *  - `config.json`        (LM architecture + asr_delay_in_tokens)
 *  - `model.safetensors`  (LM weights)
 *  - `mimi.safetensors`   (Mimi audio tokenizer)
 *  - `tokenizer.model`    (SentencePiece pieces table)
 */
object ModelAssets {

    data class Paths(
        val mimi: String,
        val lm: String,
        val tokenizer: String,
        val config: String,
    )

    private const val TAG = "ModelAssets"
    private val FILES = listOf(
        "mimi.safetensors", "model.safetensors", "tokenizer.model", "config.json",
    )

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
            lm = File(dir, "model.safetensors").absolutePath,
            tokenizer = File(dir, "tokenizer.model").absolutePath,
            config = File(dir, "config.json").absolutePath,
        )
    }
}
