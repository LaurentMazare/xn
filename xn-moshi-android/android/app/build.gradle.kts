plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "sh.gradium.xnmoshi"
    compileSdk = 35

    defaultConfig {
        applicationId = "sh.gradium.xnmoshi"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        ndk {
            // Only aarch64 — the streaming-ASR model needs FP16 NEON which is
            // Cortex-A76+ only. armeabi-v7a / armv8.0 cores are unsupported.
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    // libxn_moshi_android.so is produced by cargo (via build.sh) into
    // src/main/jniLibs/<abi>/. Model files (multi-GB) live in assets/ and are
    // copied to filesDir on first launch.
    sourceSets.getByName("main") {
        jniLibs.srcDirs("src/main/jniLibs")
    }

    androidResources {
        // Don't compress .safetensors / SP .model — they're already either
        // packed or won't gain anything, and we mmap them from filesDir.
        noCompress += listOf("safetensors", "model")
    }

    packaging {
        jniLibs.useLegacyPackaging = false
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation("androidx.activity:activity-ktx:1.9.2")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
}
