plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidLibrary)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.detekt)
}

kotlin {
    androidTarget {
        compilations.all {
            kotlinOptions {
                jvmTarget = "1.8"
            }
        }
    }

    jvm()

    sourceSets {
        commonMain.dependencies {
            implementation(libs.serialization.core)
            implementation(libs.serialization.json)
            implementation(libs.retrofit)
            implementation(libs.retrofit.serializer)
            implementation(libs.compose.ui)
            implementation(libs.compose.foundation)
            implementation(libs.compose.material)
            implementation(project.dependencies.platform(libs.compose.bom))
            // put your Multiplatform dependencies here
        }
    }
}

android {
    namespace = "ru.kheynov.hotel.shared"
    compileSdk = libs.versions.android.compileSdk.get().toInt()
    defaultConfig {
        minSdk = libs.versions.android.minSdk.get().toInt()
    }
}
