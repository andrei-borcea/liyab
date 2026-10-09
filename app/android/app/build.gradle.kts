plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    // The id (and key, below) of the earlier Java demo app, so phones that had
    // it updated in place and kept their downloaded models.
    namespace = "com.liyab.chat"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        applicationId = "com.liyab.chat"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = 28
        ndk { abiFilters += listOf("arm64-v8a") }  // libliyab is built for arm64 only
        targetSdk = flutter.targetSdkVersion
        // Uses the version code from pubspec.yaml. When using split APKs, 1000 * ABI_VERSION
        // is added automatically by Flutter. (https://developer.android.com/studio/build/configure-apk-splits#configure-APK-versions)
        // You can force using the value of versionCode by specifying the `-P force-version-code-ignoring-abi=true`
        // flag during build.
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    // libliyab.so comes from scripts/build_flutter_app.sh (CMake + NDK), copied here.
    sourceSets["main"].jniLibs.srcDir("src/main/jniLibs")

    // build/liyab-debug.keystore (scripts/build_flutter_app.sh creates it): an
    // APK signed with another key cannot update the installed app. Falls back
    // to the debug key when it does not exist.
    val liyabKeystore = rootProject.file("../../build/liyab-debug.keystore")
    signingConfigs {
        create("liyab") {
            storeFile = liyabKeystore
            storePassword = "android"
            keyAlias = "liyab"
            keyPassword = "android"
        }
    }
    buildTypes {
        val signing = if (liyabKeystore.exists()) signingConfigs.getByName("liyab") else signingConfigs.getByName("debug")
        release { signingConfig = signing }
        debug { signingConfig = signing }
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}
