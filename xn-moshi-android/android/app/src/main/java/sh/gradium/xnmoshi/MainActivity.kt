package sh.gradium.xnmoshi

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.SystemClock
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var status: TextView
    private lateinit var progress: ProgressBar
    private lateinit var dtypeSpinner: Spinner
    private lateinit var startStop: Button
    private lateinit var clearBtn: Button
    private lateinit var elapsedView: TextView
    private lateinit var transcript: TextView
    private lateinit var transcriptScroll: ScrollView

    private var asr: Asr? = null
    private var currentDtype: String = DEFAULT_DTYPE
    private var running: Boolean = false
    private var captureJob: Job? = null
    private var tickerJob: Job? = null
    private var startedAtMs: Long = 0L

    private val requestMic = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) startListening()
        else Toast.makeText(this, R.string.permission_denied, Toast.LENGTH_LONG).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        status = findViewById(R.id.status)
        progress = findViewById(R.id.progress)
        dtypeSpinner = findViewById(R.id.dtype_spinner)
        startStop = findViewById(R.id.start_stop)
        clearBtn = findViewById(R.id.clear)
        elapsedView = findViewById(R.id.elapsed)
        transcript = findViewById(R.id.transcript)
        transcriptScroll = findViewById(R.id.transcript_scroll)

        dtypeSpinner.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, DTYPES,
        )
        dtypeSpinner.setSelection(DTYPES.indexOf(DEFAULT_DTYPE).coerceAtLeast(0))

        startStop.setOnClickListener { onStartStop() }
        clearBtn.setOnClickListener { onClear() }

        // Reloading on dtype change tears down the session & rebuilds; safe
        // only while idle. Disabled mid-run via [setRunningUi].
        dtypeSpinner.onItemSelectedListener =
            object : android.widget.AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    parent: android.widget.AdapterView<*>?, view: View?,
                    pos: Int, id: Long,
                ) {
                    val picked = DTYPES[pos]
                    if (picked != currentDtype && !running) {
                        lifecycleScope.launch { reloadAsr(picked) }
                    }
                }
                override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
            }

        lifecycleScope.launch { bootstrap() }
    }

    private suspend fun bootstrap() {
        progress.visibility = View.VISIBLE
        status.text = getString(R.string.status_loading)
        try {
            val paths = withContext(Dispatchers.IO) { ModelAssets.extract(applicationContext) }
            asr = withContext(Dispatchers.Default) {
                Asr.init(paths.mimi, paths.lm, paths.tokenizer, currentDtype)
            }
            status.text = getString(R.string.status_ready)
            startStop.isEnabled = true
        } catch (e: Exception) {
            status.text = "load failed: ${e.message}"
        } finally {
            progress.visibility = View.GONE
        }
    }

    private suspend fun reloadAsr(dtype: String) {
        setRunningUi(false)
        progress.visibility = View.VISIBLE
        status.text = getString(R.string.status_loading)
        startStop.isEnabled = false
        try {
            asr?.close()
            asr = null
            val paths = withContext(Dispatchers.IO) { ModelAssets.extract(applicationContext) }
            asr = withContext(Dispatchers.Default) {
                Asr.init(paths.mimi, paths.lm, paths.tokenizer, dtype)
            }
            currentDtype = dtype
            status.text = getString(R.string.status_ready)
            startStop.isEnabled = true
        } catch (e: Exception) {
            status.text = "load failed: ${e.message}"
        } finally {
            progress.visibility = View.GONE
        }
    }

    private fun onStartStop() {
        if (running) {
            stopListening()
        } else {
            val granted = ContextCompat.checkSelfPermission(
                this, Manifest.permission.RECORD_AUDIO,
            ) == PackageManager.PERMISSION_GRANTED
            if (granted) startListening() else requestMic.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    private fun startListening() {
        val asr = this.asr ?: return
        setRunningUi(true)
        startedAtMs = SystemClock.elapsedRealtime()

        // 10 Hz UI ticker for the elapsed counter. Cheap; runs while we listen.
        tickerJob = lifecycleScope.launch {
            while (isActive && running) {
                elapsedView.text = formatElapsed(SystemClock.elapsedRealtime() - startedAtMs)
                delay(100)
            }
        }

        // Capture + inference on a dedicated background coroutine. AudioRecord
        // is created here so the lifecycle matches the mic session exactly.
        captureJob = CoroutineScope(Dispatchers.Default).launch {
            val capture = AudioCapture()
            try {
                capture.start()
                while (isActive && running) {
                    val frame = capture.readFrame() ?: break
                    val newText = try {
                        asr.step(frame)
                    } catch (e: Exception) {
                        withContext(Dispatchers.Main) {
                            status.text = "step failed: ${e.message}"
                        }
                        break
                    }
                    if (newText.isNotEmpty()) {
                        withContext(Dispatchers.Main) { appendTranscript(newText) }
                    }
                }
            } finally {
                capture.close()
            }
        }
    }

    private fun stopListening() {
        captureJob?.cancel()
        captureJob = null
        tickerJob?.cancel()
        tickerJob = null
        setRunningUi(false)
        status.text = getString(R.string.status_stopped)
    }

    private fun onClear() {
        transcript.text = ""
        elapsedView.text = formatElapsed(0)
        // Reset Rust-side accumulated tokens so the next utterance starts
        // fresh (otherwise the next "delta" includes the old prefix again on
        // the first emission after clear).
        if (!running) {
            try { asr?.reset() } catch (_: Exception) {}
        }
    }

    private fun appendTranscript(s: String) {
        transcript.append(s)
        // Auto-scroll to bottom so the most recent text stays visible.
        transcriptScroll.post { transcriptScroll.fullScroll(View.FOCUS_DOWN) }
    }

    private fun setRunningUi(isRunning: Boolean) {
        running = isRunning
        startStop.text = getString(if (isRunning) R.string.stop else R.string.start)
        dtypeSpinner.isEnabled = !isRunning
        if (isRunning) {
            status.text = getString(R.string.status_running)
        }
    }

    private fun formatElapsed(ms: Long): String {
        val tenths = (ms / 100) % 10
        val secs = (ms / 1000) % 60
        val mins = ms / 60_000
        return "%02d:%02d.%d".format(mins, secs, tenths)
    }

    override fun onDestroy() {
        super.onDestroy()
        stopListening()
        asr?.close()
        asr = null
    }

    companion object {
        // CPU-only build: BF16 / F16 / FP8 are not supported. Order roughly
        // by compute cost so q4_0 is "fastest" and f32 is "slowest/highest
        // quality". q8_0 default per spec.
        private val DTYPES = listOf(
            "q4_0", "q4_1", "q4k", "q5_0", "q5_1", "q5k",
            "q6k", "q8_0", "q8_1", "q8k", "f32",
        )
        private const val DEFAULT_DTYPE = "q8_0"
    }
}
