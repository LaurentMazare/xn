package sh.gradium.ptts

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.Spinner
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var status: TextView
    private lateinit var progress: ProgressBar
    private lateinit var textInput: EditText
    private lateinit var voiceSpinner: Spinner
    private lateinit var generate: Button
    private lateinit var statsView: TextView

    private var ptts: Ptts? = null
    private var currentVoice: String = ModelDownloader.VOICES.first()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        status = findViewById(R.id.status)
        progress = findViewById(R.id.progress)
        textInput = findViewById(R.id.text_input)
        voiceSpinner = findViewById(R.id.voice_spinner)
        generate = findViewById(R.id.generate)
        statsView = findViewById(R.id.stats)

        voiceSpinner.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, ModelDownloader.VOICES,
        )
        voiceSpinner.setSelection(0)

        generate.setOnClickListener { onGenerate() }

        lifecycleScope.launch { bootstrap() }
    }

    private suspend fun bootstrap() {
        val dl = ModelDownloader(applicationContext)
        progress.visibility = android.view.View.VISIBLE
        val job = lifecycleScope.launch {
            dl.progress.collect { p ->
                status.text = when {
                    p.label == "idle" -> "idle"
                    p.label == "done" -> "ready"
                    p.total > 0 -> "Downloading ${p.label}: " +
                            "${p.received / 1_000_000} / ${p.total / 1_000_000} MB"
                    else -> "Downloading ${p.label}…"
                }
                if (p.total > 0) {
                    progress.isIndeterminate = false
                    progress.max = 100
                    progress.progress = ((p.received * 100) / p.total).toInt()
                } else {
                    progress.isIndeterminate = true
                }
            }
        }

        val root = try {
            dl.ensureDownloaded()
        } catch (e: Exception) {
            status.text = "download failed: ${e.message}"
            progress.visibility = android.view.View.GONE
            job.cancel(); return
        }
        job.cancel()
        progress.visibility = android.view.View.GONE
        status.text = "Loading model…"
        progress.isIndeterminate = true
        progress.visibility = android.view.View.VISIBLE

        currentVoice = voiceSpinner.selectedItem as String
        ptts = withContext(Dispatchers.Default) {
            Ptts.init(root.absolutePath, currentVoice)
        }

        progress.visibility = android.view.View.GONE
        status.text = "Ready — tap Generate."
        generate.isEnabled = true
    }

    private fun onGenerate() {
        val p = ptts ?: return
        val text = textInput.text.toString().ifBlank { getString(R.string.default_text) }
        val voice = voiceSpinner.selectedItem as String

        // If the user changed voice since init, we need a fresh session since
        // the voice embedding is baked into the primed base state.
        if (voice != currentVoice) {
            generate.isEnabled = false
            status.text = "Switching voice…"
            lifecycleScope.launch {
                val root = ModelDownloader(applicationContext).root
                p.close()
                ptts = withContext(Dispatchers.Default) {
                    Ptts.init(root.absolutePath, voice)
                }
                currentVoice = voice
                generate.isEnabled = true
                doGenerate(text)
            }
            return
        }
        doGenerate(text)
    }

    private fun doGenerate(text: String) {
        val p = ptts ?: return
        generate.isEnabled = false
        statsView.text = ""
        status.text = "Generating…"
        progress.isIndeterminate = true
        progress.visibility = android.view.View.VISIBLE

        lifecycleScope.launch(Dispatchers.Default) {
            val sampleRate = p.sampleRate()
            val track = newAudioTrack(sampleRate)
            track.play()
            try {
                p.generate(text, 0.7f, 4242424242424242L)
                var framesWritten = 0
                while (true) {
                    val chunk = p.nextChunk() ?: break
                    // AudioTrack.write with ENCODING_PCM_FLOAT takes float[] directly.
                    val wrote = track.write(chunk, 0, chunk.size, AudioTrack.WRITE_BLOCKING)
                    if (wrote > 0) framesWritten += wrote // mono float: samples == frames
                }
                // Drain the hardware buffer before we stop/release. Without this
                // the tail (a few hundred ms) gets cut off.
                val deadline = System.currentTimeMillis() + 10_000
                while (track.playbackHeadPosition < framesWritten
                    && System.currentTimeMillis() < deadline
                ) {
                    Thread.sleep(20)
                }
                track.stop()
            } finally {
                track.release()
            }
            val s = p.stats()
            withContext(Dispatchers.Main) {
                progress.visibility = android.view.View.GONE
                status.text = "Done."
                generate.isEnabled = true
                statsView.text = formatStats(s)
            }
        }
    }

    private fun formatStats(s: Ptts.Stats): String = buildString {
        append("RTF             %.3f\n".format(s.rtf))
        append("avg step        %.1f ms\n".format(s.avgStepMs))
        append("audio duration  %.2f s\n".format(s.durationS))
        append("total elapsed   %.2f s\n".format(s.totalElapsedS))
        append("first audio     %.2f s\n".format(s.firstAudioS))
        append("peak RSS        %.1f MB".format(s.peakRssMb))
    }

    private fun newAudioTrack(sampleRate: Int): AudioTrack {
        val minBuf = AudioTrack.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT,
        ).coerceAtLeast(sampleRate * 4) // ~1s of audio
        return AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build(),
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build(),
            )
            .setBufferSizeInBytes(minBuf)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
    }

    override fun onDestroy() {
        super.onDestroy()
        ptts?.close()
        ptts = null
    }
}
